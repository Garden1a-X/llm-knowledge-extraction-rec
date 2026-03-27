#!/usr/bin/env python3
"""
Train MKGAT using our own data split (same as our model).

MKGAT combines knowledge graph and visual features.

Usage:
    python scripts/train_mkgat_our_split.py --seed 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))

import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

from src.data.dataset import split_data
from baselines.mkgat_model import MKGAT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KGDataLoader:
    """Load knowledge graph and build neighbor sampling structures."""

    def __init__(self, kg_file, n_entities, n_items):
        self.n_entities = n_entities
        self.n_items = n_items
        self.kg_dict = defaultdict(list)
        self.relation_dict = {}
        self._load_kg(kg_file)

    def _load_kg(self, kg_file):
        print(f"Loading KG from {kg_file}...")

        relation_id = 1
        entity_map = {}
        next_entity_id = self.n_items + 1

        with open(kg_file, 'r') as f:
            next(f)  # Skip header

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue

                head, relation, tail = parts

                try:
                    head_id = int(head)

                    if tail.isdigit():
                        tail_id = int(tail)
                    else:
                        if tail not in entity_map:
                            entity_map[tail] = next_entity_id
                            next_entity_id += 1
                            if next_entity_id > self.n_entities:
                                continue
                        tail_id = entity_map[tail]

                    if relation not in self.relation_dict:
                        self.relation_dict[relation] = relation_id
                        relation_id += 1

                    rel_id = self.relation_dict[relation]
                    self.kg_dict[head_id].append((rel_id, tail_id))

                except ValueError:
                    continue

        print(f"  Entities with neighbors: {len(self.kg_dict)}")
        print(f"  Relations: {len(self.relation_dict)}")

        self.n_relations = len(self.relation_dict)
        self.max_entity_id = max(next_entity_id - 1, max(self.kg_dict.keys()) if self.kg_dict else 0)

    def sample_neighbors(self, entity_ids, n_neighbors=8, n_layers=2):
        batch_size = len(entity_ids)
        adj_entity = np.zeros((batch_size, n_layers, n_neighbors), dtype=np.int64)
        adj_relation = np.zeros((batch_size, n_layers, n_neighbors), dtype=np.int64)

        for i, entity_id in enumerate(entity_ids):
            current_entities = [entity_id]

            for layer in range(n_layers):
                next_entities = []
                layer_relations = []

                for ent in current_entities:
                    neighbors = self.kg_dict.get(ent, [])

                    if neighbors:
                        sampled = random.choices(neighbors, k=n_neighbors)
                    else:
                        sampled = [(0, ent)] * n_neighbors

                    for rel, neighbor in sampled:
                        next_entities.append(neighbor)
                        layer_relations.append(rel)

                n_to_store = min(n_neighbors, len(next_entities))
                sampled_idx = random.sample(range(len(next_entities)), n_to_store)
                for j, idx in enumerate(sampled_idx):
                    adj_entity[i, layer, j] = next_entities[idx]
                    adj_relation[i, layer, j] = layer_relations[idx]

                current_entities = [next_entities[idx] for idx in sampled_idx]

        return adj_entity, adj_relation


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


def evaluate_uni100(model, test_data, train_pos, val_pos, n_items, n_users, kg_loader, device, k=10):
    """Evaluate with uni100 mode (1 pos + 99 neg)."""
    model.eval()

    user_pos_items = defaultdict(set)
    for u, items in train_pos.items():
        user_pos_items[int(u)].update(int(i) for i in items)
    for u, items in val_pos.items():
        user_pos_items[int(u)].update(int(i) for i in items)

    # Pre-compute all item embeddings with KG aggregation
    print("  Pre-computing item embeddings with KG aggregation...")
    all_item_embeddings = []

    with torch.no_grad():
        batch_size = 512
        dummy_user = torch.LongTensor([1]).to(device)

        for i in tqdm(range(0, n_items, batch_size), desc="  Items", leave=False):
            batch_items = list(range(i+1, min(i+batch_size+1, n_items+1)))
            if not batch_items:
                continue

            adj_entity, adj_relation = kg_loader.sample_neighbors(batch_items, n_layers=model.n_layers)
            adj_entity = torch.LongTensor(adj_entity).to(device)
            adj_relation = torch.LongTensor(adj_relation).to(device)

            batch_items_t = torch.LongTensor(batch_items).to(device)
            dummy_users = dummy_user.repeat(len(batch_items))

            _, item_emb = model.forward(dummy_users, batch_items_t, adj_entity, adj_relation)
            all_item_embeddings.append(item_emb.cpu())

        all_item_embeddings = torch.cat(all_item_embeddings, dim=0)

        # Pre-compute user embeddings
        all_user_embeddings = model.user_embed.weight[1:n_users+1].cpu()

    recalls = []
    ndcgs = []
    hits = []

    print("  Evaluating...")
    for user, pos_item in tqdm(test_data, desc="  Scoring", leave=False):
        neg_items = []
        while len(neg_items) < 99:
            neg = random.randint(1, n_items)
            if neg not in user_pos_items[user] and neg != pos_item and neg not in neg_items:
                neg_items.append(neg)

        candidates = [pos_item] + neg_items

        user_emb = all_user_embeddings[user - 1]
        item_embs = all_item_embeddings[[i - 1 for i in candidates]]

        scores = (user_emb @ item_embs.T).numpy()

        ranked = np.argsort(-scores)
        pos_rank = np.where(ranked == 0)[0][0]

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
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='data/recbole/amazon-beauty')
    parser.add_argument('--visual_features', type=str, default='data/recbole/amazon-beauty/visual_features.npy')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("MKGAT with Our Data Split")
    print("="*60)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Layers: {args.n_layers}")
    print()

    # Load and split data using OUR split function
    inter_path = f'{args.data_path}/amazon-beauty.inter'
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

    # Load KG
    kg_file = f'{args.data_path}/amazon-beauty.kg'
    n_entities_est = n_items * 20
    kg_loader = KGDataLoader(kg_file, n_entities_est, n_items)
    n_relations = kg_loader.n_relations
    print(f"KG loaded: {n_relations} relations")

    # Load visual features
    print(f"\nLoading visual features from {args.visual_features}...")
    visual_features = np.load(args.visual_features)
    print(f"Visual features shape: {visual_features.shape}")
    visual_dim = visual_features.shape[1]

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
    print("\nBuilding MKGAT model...")
    model = MKGAT(
        n_users=n_users,
        n_items=n_items,
        n_entities=n_entities_est,
        n_relations=n_relations,
        embedding_dim=args.embedding_dim,
        visual_dim=visual_dim,
        n_layers=args.n_layers,
        aggregator_type='bi-interaction',
        dropout=0.2,
        reg_weight=1e-4
    ).to(device)

    model.load_visual_features(visual_features)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    best_val_ndcg = 0
    patience = 0
    best_epoch = 0

    Path('outputs/beauty').mkdir(parents=True, exist_ok=True)

    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for users, pos_items, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            # Sample KG neighbors
            all_items_batch = torch.cat([pos_items, neg_items])
            adj_entity, adj_relation = kg_loader.sample_neighbors(
                all_items_batch.numpy().tolist(), n_layers=args.n_layers
            )

            batch_size = users.shape[0]
            adj_entity_pos = torch.LongTensor(adj_entity[:batch_size]).to(device)
            adj_entity_neg = torch.LongTensor(adj_entity[batch_size:]).to(device)
            adj_relation_pos = torch.LongTensor(adj_relation[:batch_size]).to(device)
            adj_relation_neg = torch.LongTensor(adj_relation[batch_size:]).to(device)

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()

            pos_scores = model.predict(users, pos_items, adj_entity_pos, adj_relation_pos)
            neg_scores = model.predict(users, neg_items, adj_entity_neg, adj_relation_neg)

            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            reg_loss = model.get_reg_loss(users, pos_items)
            loss = bpr_loss + reg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        val_metrics = evaluate_uni100(model, val_data, train_pos, {}, n_items, n_users, kg_loader, device)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val NDCG@10={val_metrics['ndcg@10']:.4f}, Val Recall@10={val_metrics['recall@10']:.4f}")

        if val_metrics['ndcg@10'] > best_val_ndcg:
            best_val_ndcg = val_metrics['ndcg@10']
            best_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), 'outputs/beauty/mkgat_our_split_best.pth')
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load('outputs/beauty/mkgat_our_split_best.pth'))

    print("Testing...")
    test_metrics = evaluate_uni100(model, test_data, train_pos, val_pos, n_items, n_users, kg_loader, device)

    print()
    print("="*60)
    print("Test Results (MKGAT with Our Split)")
    print("="*60)
    print(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"  Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  Hit@10: {test_metrics['hit@10']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
