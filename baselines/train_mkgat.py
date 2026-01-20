#!/usr/bin/env python3
"""
Train MKGAT model with visual features.

Supports both MovieLens 1M and Amazon Video Games datasets.

Usage:
    python baselines/train_mkgat.py \
        --data_dir data/recbole/ml-1m \
        --visual_features data/recbole/ml-1m/visual_features.npy \
        --output_dir outputs/mkgat \
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

from mkgat_model import MKGAT


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


class KGDataLoader:
    """Load knowledge graph and build neighbor sampling structures."""

    def __init__(self, kg_file, n_entities, n_items):
        """
        Args:
            kg_file: Path to .kg file
            n_entities: Total number of entities (for embedding size)
            n_items: Number of items (to allocate entity IDs after item IDs)
        """
        self.n_entities = n_entities
        self.n_items = n_items
        self.kg_dict = defaultdict(list)  # entity -> [(relation, tail_entity)]
        self.relation_dict = {}  # relation_name -> relation_id

        self._load_kg(kg_file)

    def _load_kg(self, kg_file):
        """Load KG triplets and build neighbor dictionary."""
        print(f"Loading KG from {kg_file}...")

        relation_id = 1  # Start from 1 for RecBole compatibility
        entity_map = {}  # Map entity names to IDs
        next_entity_id = self.n_items + 1  # Start after item IDs (items are 1-indexed)

        with open(kg_file, 'r') as f:
            next(f)  # Skip header

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue

                head, relation, tail = parts

                # Map head and tail to entity IDs
                try:
                    head_id = int(head)

                    # Tail might be an entity name or item ID
                    if tail.isdigit():
                        tail_id = int(tail)
                    else:
                        # Create entity ID for non-item entities using a mapping dict
                        if tail not in entity_map:
                            entity_map[tail] = next_entity_id
                            next_entity_id += 1
                            # Safety check
                            if next_entity_id > self.n_entities:
                                print(f"Warning: Entity ID exceeded limit, skipping {tail}")
                                continue
                        tail_id = entity_map[tail]

                    # Map relation to ID
                    if relation not in self.relation_dict:
                        self.relation_dict[relation] = relation_id
                        relation_id += 1

                    rel_id = self.relation_dict[relation]

                    # Add to KG dictionary
                    self.kg_dict[head_id].append((rel_id, tail_id))

                except ValueError:
                    continue

        print(f"✓ Loaded KG:")
        print(f"  Entities with neighbors: {len(self.kg_dict)}")
        print(f"  Total entities (including attributes): {len(entity_map) + len(self.kg_dict)}")
        print(f"  Relations: {len(self.relation_dict)}")

        self.n_relations = len(self.relation_dict)
        self.max_entity_id = max(next_entity_id - 1, max(self.kg_dict.keys()) if self.kg_dict else 0)

    def sample_neighbors(self, entity_ids, n_neighbors=8, n_layers=3):
        """
        Sample neighbors for given entities.

        Args:
            entity_ids: List of entity IDs
            n_neighbors: Number of neighbors to sample per layer
            n_layers: Number of hop layers

        Returns:
            adj_entity: (batch_size, n_layers, n_neighbors) neighbor entity IDs
            adj_relation: (batch_size, n_layers, n_neighbors) relation IDs
        """
        batch_size = len(entity_ids)
        adj_entity = np.zeros((batch_size, n_layers, n_neighbors), dtype=np.int64)
        adj_relation = np.zeros((batch_size, n_layers, n_neighbors), dtype=np.int64)

        for i, entity_id in enumerate(entity_ids):
            # Sample neighbors for each layer
            current_entities = [entity_id]

            for layer in range(n_layers):
                next_entities = []
                layer_relations = []

                for ent in current_entities:
                    neighbors = self.kg_dict.get(ent, [])

                    if neighbors:
                        # Sample neighbors with replacement
                        sampled = random.choices(neighbors, k=n_neighbors)
                    else:
                        # If no neighbors, use self-loop (relation=0 for padding)
                        sampled = [(0, ent)] * n_neighbors

                    for rel, neighbor in sampled:
                        next_entities.append(neighbor)
                        layer_relations.append(rel)

                # Store sampled neighbors (take first n_neighbors if we have multiple entities)
                # For first layer: we have n_neighbors from single entity
                # For later layers: we have n_neighbors * len(current_entities)
                n_to_store = min(n_neighbors, len(next_entities))
                sampled_idx = random.sample(range(len(next_entities)), n_to_store)
                for j, idx in enumerate(sampled_idx):
                    adj_entity[i, layer, j] = next_entities[idx]
                    adj_relation[i, layer, j] = layer_relations[idx]

                # Update current entities for next layer: only use the selected neighbors
                current_entities = [next_entities[idx] for idx in sampled_idx]

        return adj_entity, adj_relation


class InteractionDataset(Dataset):
    """User-item interaction dataset."""

    def __init__(self, interactions, neg_sampling=True, n_items=None):
        """
        Args:
            interactions: List of (user, item, rating) tuples
            neg_sampling: Whether to do negative sampling
            n_items: Number of items (for negative sampling)
        """
        self.interactions = interactions
        self.neg_sampling = neg_sampling
        self.n_items = n_items

        if neg_sampling:
            # Build user -> positive items mapping for negative sampling
            # RecBole treats ALL interactions as positive (implicit feedback)
            self.user_pos_items = defaultdict(set)
            for user, item, rating in interactions:
                self.user_pos_items[user].add(item)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item, rating = self.interactions[idx]

        # RecBole treats ALL interactions as positive (implicit feedback)
        # No rating threshold filtering
        label = 1.0

        if self.neg_sampling:
            # Sample negative item
            while True:
                neg_item = random.randint(1, self.n_items)
                if neg_item not in self.user_pos_items[user]:
                    break
            return user, item, neg_item, label
        else:
            return user, item, item, label  # No negative sampling


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
    This matches RecBole's RS (Ratio Split) + TO (Time Ordering) mode.
    """
    # Group interactions by user
    user_interactions = defaultdict(list)
    for u, i, r, t in interactions:
        user_interactions[u].append((u, i, r, t))

    # Split each user's interactions temporally
    train = []
    val = []
    test = []

    for user, user_inters in user_interactions.items():
        # Sort by timestamp for this user
        user_inters.sort(key=lambda x: x[3])

        n = len(user_inters)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Split for this user
        train.extend([(u, i, r) for u, i, r, t in user_inters[:train_end]])
        val.extend([(u, i, r) for u, i, r, t in user_inters[train_end:val_end]])
        test.extend([(u, i, r) for u, i, r, t in user_inters[val_end:]])

    return train, val, test


def evaluate(model, dataloader, kg_loader, device, k=10, n_items=None, mode='uni100',
             train_data=None, val_data=None, n_layers=2):
    """
    Evaluate model with Recall@K, NDCG@K, Precision@K using uni100 mode.

    Args:
        model: MKGAT model
        dataloader: Evaluation dataloader (contains positive items only)
        kg_loader: KG data loader for neighbor sampling
        device: torch device
        k: Top-K
        n_items: Total number of items (for negative sampling)
        mode: 'uni100' (1 pos + 99 neg) or 'full' (all items)
        train_data: Training data (list of (user, item, rating)) for negative sampling
        val_data: Validation data (optional, for test evaluation)
        n_layers: Number of GNN layers for neighbor sampling

    Returns:
        Dictionary of metrics
    """
    model.eval()

    # Pre-compute ALL item embeddings WITH KG aggregation
    print("  Pre-computing item embeddings with KG aggregation...")
    all_item_embeddings = []
    all_user_embeddings = []

    with torch.no_grad():
        # Compute item embeddings for all items (1 to n_items)
        # Use dummy user (user_id=1) to compute item embeddings through forward pass
        batch_size_precompute = 512
        dummy_user = torch.LongTensor([1]).to(device)

        for i in tqdm(range(0, n_items, batch_size_precompute), desc="  Items", leave=False):
            batch_items = list(range(i+1, min(i+batch_size_precompute+1, n_items+1)))
            if not batch_items:
                continue

            # Sample KG for this batch of items
            adj_entity, adj_relation = kg_loader.sample_neighbors(batch_items, n_layers=n_layers)
            adj_entity = torch.LongTensor(adj_entity).to(device)
            adj_relation = torch.LongTensor(adj_relation).to(device)

            batch_items_t = torch.LongTensor(batch_items).to(device)

            # Use forward pass to get KG-aggregated item embeddings
            # Repeat dummy user for batch
            dummy_users = dummy_user.repeat(len(batch_items))
            _, item_emb = model.forward(dummy_users, batch_items_t, adj_entity, adj_relation)
            all_item_embeddings.append(item_emb.cpu())

        # Stack all item embeddings (n_items, embedding_dim)
        all_item_embeddings = torch.cat(all_item_embeddings, dim=0)

        # Compute user embeddings
        for user_id in range(1, model.n_users + 1):
            user_emb = model.user_embed(torch.LongTensor([user_id]).to(device))
            all_user_embeddings.append(user_emb.cpu())

        all_user_embeddings = torch.cat(all_user_embeddings, dim=0)

    # Collect all interactions to evaluate (RecBole treats all as positive)
    pos_interactions = []
    with torch.no_grad():
        for batch in dataloader:
            users, items, _, labels = batch
            for u, i in zip(users.numpy(), items.numpy()):
                pos_interactions.append((u, i))

    # Build user positive items for negative sampling
    # Must exclude ALL historical interactions (train + val for test eval)
    user_pos_items = defaultdict(set)

    # Add training interactions
    if train_data:
        for u, i, r in train_data:
            user_pos_items[u].add(i)

    # Add validation interactions (only when evaluating test set)
    if val_data:
        for u, i, r in val_data:
            user_pos_items[u].add(i)

    # Add current evaluation interactions
    for u, i in pos_interactions:
        user_pos_items[u].add(i)

    # Evaluate using pre-computed embeddings (super fast!)
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

        # Get user embedding
        user_emb = all_user_embeddings[user - 1]  # 0-indexed

        # Get item embeddings (0-indexed)
        item_embs = all_item_embeddings[[i - 1 for i in candidate_items]]

        # Compute scores via dot product (NO KG sampling needed!)
        scores = (user_emb @ item_embs.T).numpy()

        # Rank by scores
        ranked_indices = np.argsort(-scores)  # Descending
        pos_rank = np.where(ranked_indices == 0)[0][0]  # Position of pos item (index 0)

        # Recall@K
        recall = 1.0 if pos_rank < k else 0.0
        recalls.append(recall)

        # Precision@K
        precision = 1.0 / k if pos_rank < k else 0.0
        precisions.append(precision)

        # NDCG@K
        if pos_rank < k:
            ndcg = 1.0 / np.log2(pos_rank + 2)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)

    return {
        f'recall@{k}': np.mean(recalls) if recalls else 0,
        f'ndcg@{k}': np.mean(ndcgs) if ndcgs else 0,
        f'precision@{k}': np.mean(precisions) if precisions else 0
    }


def train_epoch(model, dataloader, kg_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        users, pos_items, neg_items, _ = batch

        # Sample KG neighbors for positive and negative items
        all_items = torch.cat([pos_items, neg_items])
        adj_entity, adj_relation = kg_loader.sample_neighbors(all_items.numpy().tolist(), n_layers=model.n_layers)

        batch_size = users.shape[0]
        adj_entity_pos = adj_entity[:batch_size]
        adj_entity_neg = adj_entity[batch_size:]
        adj_relation_pos = adj_relation[:batch_size]
        adj_relation_neg = adj_relation[batch_size:]

        # Move to device
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        adj_entity_pos = torch.LongTensor(adj_entity_pos).to(device)
        adj_entity_neg = torch.LongTensor(adj_entity_neg).to(device)
        adj_relation_pos = torch.LongTensor(adj_relation_pos).to(device)
        adj_relation_neg = torch.LongTensor(adj_relation_neg).to(device)

        # Predict scores
        pos_scores = model.predict(users, pos_items, adj_entity_pos, adj_relation_pos)
        neg_scores = model.predict(users, neg_items, adj_entity_neg, adj_relation_neg)

        # BPR loss: positive items should rank higher than negative items
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

        # Regularization
        reg_loss = model.get_reg_loss(users, pos_items)

        # Total loss
        loss = bpr_loss + reg_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train MKGAT with visual features')

    # Data args
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .inter, .kg, .item files')
    parser.add_argument('--visual_features', type=str, required=True,
                        help='Path to visual features .npy file')

    # Model args
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of aggregation layers (default: 2 to align with Ours)')
    parser.add_argument('--aggregator_type', type=str, default='bi-interaction',
                        choices=['bi-interaction', 'gcn', 'graphsage'],
                        help='Aggregator type')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (aligned with Ours method)')
    parser.add_argument('--reg_weight', type=float, default=1e-4,
                        help='L2 regularization weight (aligned with KGAT)')

    # Training args
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size (default: 2048 for speed)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluate every N epochs (default: 1 to match RecBole)')

    # Other args
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/mkgat',
                        help='Output directory')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Detect dataset name from data_dir
    data_dir = Path(args.data_dir)
    dataset_name = data_dir.name  # e.g., 'ml-1m' or 'amazon-videogames'

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"seed_{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Training MKGAT on {dataset_name}")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # Load data
    data_dir = Path(args.data_dir)
    inter_file = data_dir / f"{data_dir.name}.inter"
    kg_file = data_dir / f"{data_dir.name}.kg"
    item_file = data_dir / f"{data_dir.name}.item"

    print("Loading interactions...")
    interactions = load_interactions(inter_file)
    print(f"✓ Loaded {len(interactions)} interactions")

    # Get data stats
    users = set(u for u, _, _, _ in interactions)
    items = set(i for _, i, _, _ in interactions)
    n_users = max(users)
    n_items = max(items)
    print(f"✓ Users: {n_users}, Items: {n_items}")
    print()

    # Split data temporally
    print("Splitting data (70/10/20, temporal)...")
    train_data, val_data, test_data = split_data_temporal(interactions)
    print(f"✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print()

    # Load KG
    # Estimate n_entities conservatively as 20x n_items (items + KG entities)
    # Video Games may have many category entities, so use larger multiplier
    n_entities_est = n_items * 20
    kg_loader = KGDataLoader(kg_file, n_entities_est, n_items)
    n_relations = kg_loader.n_relations
    print(f"✓ KG loaded: {n_relations} relations")
    print()

    # Create datasets
    train_dataset = InteractionDataset(train_data, neg_sampling=True, n_items=n_items)
    val_dataset = InteractionDataset(val_data, neg_sampling=False)
    test_dataset = InteractionDataset(test_data, neg_sampling=False)

    # Use multiple workers for parallel data loading
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Load visual features
    print(f"Loading visual features from {args.visual_features}...")
    visual_features = np.load(args.visual_features)
    print(f"✓ Visual features shape: {visual_features.shape}")
    print()

    # Create model
    print("Creating MKGAT model...")
    model = MKGAT(
        n_users=n_users,
        n_items=n_items,
        n_entities=n_entities_est,
        n_relations=n_relations,
        embedding_dim=args.embedding_dim,
        visual_dim=2048,
        n_layers=args.n_layers,
        aggregator_type=args.aggregator_type,
        dropout=args.dropout,
        reg_weight=args.reg_weight
    )
    model.load_visual_features(visual_features)
    model.to(device)

    print(f"✓ Model created:")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Aggregator: {args.aggregator_type}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    print()

    best_val_ndcg = 0
    patience_counter = 0
    train_losses = []
    val_metrics_history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, kg_loader, optimizer, device)
        train_losses.append(train_loss)

        # Evaluate only every N epochs (for speed)
        should_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs)

        if should_eval:
            # Evaluate on validation (uni100 mode)
            # Only exclude train data when evaluating val
            val_metrics = evaluate(model, val_loader, kg_loader, device, k=10, n_items=n_items,
                                 train_data=train_data, n_layers=args.n_layers)

            print(f"Epoch {epoch}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val NDCG@10: {val_metrics['ndcg@10']:.4f}")
            print(f"  Val Recall@10: {val_metrics['recall@10']:.4f}")

            val_metrics_history.append(val_metrics)

            # Early stopping
            if val_metrics['ndcg@10'] > best_val_ndcg:
                best_val_ndcg = val_metrics['ndcg@10']
                patience_counter = 0

                # Save best model
                torch.save(model.state_dict(), output_dir / 'best_model.pth')
                print(f"  ✓ New best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= args.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            print()
        else:
            # Just print training loss
            print(f"Epoch {epoch}/{args.epochs}: Train Loss = {train_loss:.4f}")

    # Load best model and evaluate on test
    print("Evaluating best model on test set (uni100 mode)...")
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    # Exclude both train and val data when evaluating test
    test_metrics = evaluate(model, test_loader, kg_loader, device, k=10, n_items=n_items,
                          train_data=train_data, val_data=val_data, n_layers=args.n_layers)

    print()
    print("="*80)
    print("Test Results:")
    print("="*80)
    print(f"NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"Precision@10: {test_metrics['precision@10']:.4f}")
    print("="*80)

    # Save results
    results = {
        'args': vars(args),
        'test_metrics': test_metrics,
        'best_val_metrics': val_metrics_history[np.argmax([m['ndcg@10'] for m in val_metrics_history])],
        'train_losses': train_losses
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == '__main__':
    main()
