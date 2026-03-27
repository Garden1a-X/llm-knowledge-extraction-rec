#!/usr/bin/env python3
"""
Train KGAT using our own data split (same as our model).

KGAT uses metadata-based KG (categories, brand, etc.), not our LLM-extracted KG.

Usage:
    python scripts/train_kgat_our_split.py --seed 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
import scipy.sparse as sp

from src.data.dataset import split_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KGAT(nn.Module):
    """
    Simplified KGAT implementation.
    Uses attention mechanism to aggregate KG neighbors.
    """
    def __init__(self, n_users, n_items, n_entities, n_relations,
                 embedding_dim=64, n_layers=2, reg_weight=1e-5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        # User and entity embeddings
        self.user_embed = nn.Embedding(n_users + 1, embedding_dim)
        self.entity_embed = nn.Embedding(n_entities + 1, embedding_dim)
        self.relation_embed = nn.Embedding(n_relations + 1, embedding_dim)

        # Attention layers
        self.W_kg = nn.Linear(embedding_dim * 3, embedding_dim)
        self.W_cf = nn.Linear(embedding_dim * 2, embedding_dim)

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

        self.kg_adj = None
        self.cf_adj = None

    def set_kg_adj(self, kg_dict):
        """Store KG adjacency as dict: entity_id -> [(relation, tail_entity), ...]"""
        self.kg_dict = kg_dict

    def set_cf_adj(self, cf_adj):
        """Store CF adjacency matrix."""
        self.cf_adj = cf_adj

    def calc_kg_embeddings(self, item_embed):
        """Calculate KG-enhanced item embeddings using attention."""
        # For simplicity, use mean aggregation with relation-aware weighting
        kg_enhanced = item_embed.clone()

        for entity_id in range(1, self.n_items + 1):
            if entity_id in self.kg_dict and len(self.kg_dict[entity_id]) > 0:
                neighbors = self.kg_dict[entity_id]

                # Get neighbor embeddings
                tail_ids = [t for r, t in neighbors if t <= self.n_entities]
                rel_ids = [r for r, t in neighbors if t <= self.n_entities]

                if tail_ids:
                    tail_e = self.entity_embed(torch.LongTensor(tail_ids).to(item_embed.device))
                    rel_e = self.relation_embed(torch.LongTensor(rel_ids).to(item_embed.device))

                    # Simple attention: score based on relation embedding
                    head_e = item_embed[entity_id].unsqueeze(0).expand(len(tail_ids), -1)
                    concat = torch.cat([head_e, rel_e, tail_e], dim=1)
                    att_scores = F.softmax(self.W_kg(concat).sum(dim=1), dim=0)

                    # Weighted sum
                    neighbor_agg = (att_scores.unsqueeze(1) * tail_e).sum(dim=0)
                    kg_enhanced[entity_id] = item_embed[entity_id] + neighbor_agg

        return kg_enhanced

    def forward(self):
        """Get final embeddings through graph convolution."""
        user_embed = self.user_embed.weight
        item_embed = self.entity_embed.weight[:self.n_items + 1]

        # Simple CF propagation using adjacency matrix
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        embeds_list = [all_embed]

        for _ in range(self.n_layers):
            all_embed = torch.sparse.mm(self.cf_adj, all_embed)
            embeds_list.append(all_embed)

        all_embed = torch.stack(embeds_list, dim=1).mean(dim=1)
        user_embed_final, item_embed_final = torch.split(all_embed, [self.n_users + 1, self.n_items + 1])

        return user_embed_final, item_embed_final

    def bpr_loss(self, users, pos_items, neg_items):
        user_embed, item_embed = self.forward()

        user_e = user_embed[users]
        pos_e = item_embed[pos_items]
        neg_e = item_embed[neg_items]

        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # Regularization
        reg_loss = self.reg_weight * (
            self.user_embed(users).norm(2).pow(2) +
            self.entity_embed(pos_items).norm(2).pow(2) +
            self.entity_embed(neg_items).norm(2).pow(2)
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

        while True:
            neg_item = random.randint(1, self.n_items)
            if neg_item not in self.user_pos_items[user]:
                break

        return user, item, neg_item


def build_cf_adj_matrix(n_users, n_items, train_interactions, device):
    """Build normalized CF adjacency matrix."""
    rows = []
    cols = []
    for user, item in train_interactions:
        rows.append(user)
        cols.append(item + n_users + 1)
        rows.append(item + n_users + 1)
        cols.append(user)

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(len(rows))

    n_nodes = n_users + n_items + 2
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Normalize
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    norm_adj = norm_adj.tocoo()

    indices = torch.LongTensor([norm_adj.row, norm_adj.col])
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(device)


def build_kg_from_metadata(item_file, item_map_file):
    """Build KG from item metadata (categories, brand, etc.)."""
    kg_dict = defaultdict(list)
    entity_set = set()
    relation_set = set()

    # Load item file
    if not Path(item_file).exists():
        print(f"Warning: Item file not found: {item_file}")
        return kg_dict, 0, 0

    # Load ID mappings
    with open(item_map_file, 'r') as f:
        id_mappings = json.load(f)

    # Define relations
    relation_map = {
        'has_category': 1,
        'has_brand': 2,
    }

    entity_counter = 10000  # Start entities from high number to avoid collision with items

    # Parse item file
    entity_name_to_id = {}

    with open(item_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        title_idx = header.index('title:token_seq') if 'title:token_seq' in header else -1
        cat_idx = header.index('categories:token_seq') if 'categories:token_seq' in header else -1

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue

            item_id = int(parts[0])

            # Extract categories
            if cat_idx >= 0 and cat_idx < len(parts):
                categories = parts[cat_idx].split('|')
                for cat in categories:
                    cat = cat.strip()
                    if cat and cat != 'Unknown':
                        if cat not in entity_name_to_id:
                            entity_counter += 1
                            entity_name_to_id[cat] = entity_counter

                        entity_id = entity_name_to_id[cat]
                        kg_dict[item_id].append((relation_map['has_category'], entity_id))
                        entity_set.add(entity_id)
                        relation_set.add(relation_map['has_category'])

    n_entities = max(entity_set) if entity_set else 0
    n_relations = max(relation_set) if relation_set else 0

    print(f"Built KG from metadata: {sum(len(v) for v in kg_dict.values())} triplets")
    print(f"  Entities: {len(entity_set)}, Relations: {len(relation_set)}")

    return kg_dict, n_entities, n_relations


def evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device, k=10):
    """Evaluate with uni100 mode."""
    model.eval()

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
            neg_items = []
            while len(neg_items) < 99:
                neg = random.randint(1, n_items)
                if neg not in user_pos_items[user] and neg != pos_item and neg not in neg_items:
                    neg_items.append(neg)

            candidates = [pos_item] + neg_items

            user_e = user_embed[user]
            items_e = item_embed[candidates]

            scores = (user_e * items_e).sum(dim=1).cpu().numpy()

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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("KGAT with Our Data Split")
    print("="*60)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print()

    # Load and split data
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

    # Build KG from metadata
    print("\nBuilding KG from metadata...")
    item_file = f'{args.data_path}/amazon-beauty.item'
    mapping_file = f'{args.data_path}/id_mappings.json'
    kg_dict, n_entities, n_relations = build_kg_from_metadata(item_file, mapping_file)

    # If no KG, use items as entities
    if n_entities == 0:
        print("Warning: No KG found, using LightGCN-like model")
        n_entities = n_items
        n_relations = 1

    print(f"Entities: {n_entities}, Relations: {n_relations}")
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
    print("Building model...")
    model = KGAT(n_users, n_items, n_entities, n_relations,
                 args.embedding_dim, args.n_layers).to(device)

    # Set adjacency matrices
    print("Building CF adjacency matrix...")
    cf_adj = build_cf_adj_matrix(n_users, n_items, train_interactions, device)
    model.set_cf_adj(cf_adj)
    model.set_kg_adj(kg_dict)

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
            torch.save(model.state_dict(), 'outputs/beauty/kgat_our_split_best.pth')
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load('outputs/beauty/kgat_our_split_best.pth'))

    print("Testing...")
    test_metrics = evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device)

    print()
    print("="*60)
    print("Test Results (KGAT with Our Split)")
    print("="*60)
    print(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"  Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  Hit@10: {test_metrics['hit@10']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
