#!/usr/bin/env python3
"""
MKGAT: Multi-modal Knowledge Graph Attention Network (Simplified Version)

Simplified implementation that combines:
1. Visual features from ResNet50
2. Knowledge graph embeddings with attention
3. User-item graph convolution

This version uses a simpler interface compatible with the training script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix, vstack, hstack
from collections import defaultdict


class MKGAT(nn.Module):
    """
    Simplified Multi-modal Knowledge Graph Attention Network.

    Combines:
    - MMGCN-style user-item graph convolution
    - KG-enhanced item embeddings with attention
    - Visual feature integration
    """

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        visual_dim=2048,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        reg_weight=1e-5
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            visual_dim: Dimension of visual features (ResNet50: 2048)
            n_layers: Number of GCN layers
            n_heads: Number of attention heads (for KG aggregation)
            dropout: Dropout rate
            reg_weight: L2 regularization weight
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.visual_dim = visual_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.reg_weight = reg_weight

        # User embeddings (RecBole 1-indexed)
        self.user_embed = nn.Embedding(n_users + 1, embedding_dim)

        # Item embeddings (will be enhanced with KG and visual)
        self.item_embed = nn.Embedding(n_items + 1, embedding_dim)

        # Visual feature projection
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Multi-modal fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Linear(embedding_dim * 3, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

        # Placeholders
        self.visual_features = None
        self.adj_matrix = None
        self.kg_embeddings = None  # Pre-computed KG-enhanced embeddings

    def load_visual_features(self, visual_features):
        """Load precomputed visual features."""
        self.visual_features = torch.FloatTensor(visual_features)

    def build_graph(self, interaction_matrix, kg_triplets, device):
        """
        Build graph structures for GCN propagation.

        Args:
            interaction_matrix: scipy sparse matrix of user-item interactions
            kg_triplets: list of (item_id, relation, tail) tuples from KG
            device: torch device
        """
        print("Building MKGAT graph structures...")

        # 1. Build user-item adjacency matrix (like MMGCN)
        self._build_ui_adjacency(interaction_matrix, device)

        # 2. Build KG neighbor structure for items
        self._build_kg_structure(kg_triplets, device)

        print("✓ MKGAT graph built")

    def _build_ui_adjacency(self, interaction_matrix, device):
        """Build normalized user-item adjacency matrix."""
        R = interaction_matrix.tocoo()

        # Bipartite adjacency matrix
        # A = [[0, R],
        #      [R^T, 0]]
        top = hstack([coo_matrix((self.n_users, self.n_users)), R])
        bottom = hstack([R.T, coo_matrix((self.n_items, self.n_items))])
        A = vstack([top, bottom])

        # Normalize: D^(-1/2) * A * D^(-1/2)
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv_sqrt = np.power(degrees, -0.5)
        D_inv_sqrt = coo_matrix((D_inv_sqrt, (np.arange(len(D_inv_sqrt)), np.arange(len(D_inv_sqrt)))))

        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        A_norm_coo = A_norm.tocoo()

        indices = torch.LongTensor([A_norm_coo.row, A_norm_coo.col])
        values = torch.FloatTensor(A_norm_coo.data)
        shape = A_norm_coo.shape

        self.adj_matrix = torch.sparse.FloatTensor(indices, values, torch.Size(shape)).to(device)
        print(f"  UI adjacency matrix: {shape}")

    def _build_kg_structure(self, kg_triplets, device):
        """
        Build KG neighbor structure.

        Stores neighbors for each item to compute KG-enhanced embeddings.
        """
        # Build item -> neighbors mapping
        self.item_neighbors = defaultdict(list)  # item_id -> [(relation_id, tail_id), ...]

        # Build relation and entity vocabularies
        relations = set()
        tails = set()

        for head, rel, tail in kg_triplets:
            relations.add(rel)
            tails.add(tail)

        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.tail2id = {t: i for i, t in enumerate(sorted(tails))}
        self.n_relations = len(self.relation2id)
        self.n_tails = len(self.tail2id)

        print(f"  KG: {len(kg_triplets)} triplets, {self.n_relations} relations, {self.n_tails} tail entities")

        # Relation and tail embeddings
        self.relation_embed = nn.Embedding(self.n_relations + 1, self.embedding_dim).to(device)
        self.tail_embed = nn.Embedding(self.n_tails + 1, self.embedding_dim).to(device)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.tail_embed.weight)

        # Build neighbor list for each item
        for head, rel, tail in kg_triplets:
            if 1 <= head <= self.n_items:
                rel_id = self.relation2id.get(rel, 0)
                tail_id = self.tail2id.get(tail, 0)
                self.item_neighbors[head].append((rel_id, tail_id))

        # Convert to tensors for batch processing
        max_neighbors = max(len(v) for v in self.item_neighbors.values()) if self.item_neighbors else 1
        max_neighbors = min(max_neighbors, 32)  # Limit for efficiency

        # Pad neighbor arrays
        self.neighbor_relations = torch.zeros(self.n_items + 1, max_neighbors, dtype=torch.long, device=device)
        self.neighbor_tails = torch.zeros(self.n_items + 1, max_neighbors, dtype=torch.long, device=device)
        self.neighbor_mask = torch.zeros(self.n_items + 1, max_neighbors, device=device)

        for item_id, neighbors in self.item_neighbors.items():
            if item_id > self.n_items:
                continue
            n = min(len(neighbors), max_neighbors)
            for j, (rel_id, tail_id) in enumerate(neighbors[:n]):
                self.neighbor_relations[item_id, j] = rel_id
                self.neighbor_tails[item_id, j] = tail_id
                self.neighbor_mask[item_id, j] = 1.0

        print(f"  Max neighbors per item: {max_neighbors}")

    def get_kg_enhanced_item_embed(self, item_ids, device):
        """
        Get KG-enhanced item embeddings using attention.

        Args:
            item_ids: (batch_size,) item IDs

        Returns:
            KG-enhanced embeddings (batch_size, embedding_dim)
        """
        batch_size = item_ids.size(0)

        # Base item embeddings
        item_emb = self.item_embed(item_ids)  # (batch, dim)

        # Get neighbor info
        neighbor_rels = self.neighbor_relations[item_ids]  # (batch, max_neighbors)
        neighbor_tails = self.neighbor_tails[item_ids]  # (batch, max_neighbors)
        mask = self.neighbor_mask[item_ids]  # (batch, max_neighbors)

        # Get embeddings
        rel_emb = self.relation_embed(neighbor_rels)  # (batch, max_neighbors, dim)
        tail_emb = self.tail_embed(neighbor_tails)  # (batch, max_neighbors, dim)

        # Attention: score = item_emb · (rel_emb * tail_emb)
        neighbor_emb = rel_emb * tail_emb  # (batch, max_neighbors, dim)
        item_emb_exp = item_emb.unsqueeze(1)  # (batch, 1, dim)
        scores = torch.sum(item_emb_exp * neighbor_emb, dim=-1)  # (batch, max_neighbors)

        # Apply mask
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)  # (batch, max_neighbors)
        attention = attention.masked_fill(mask == 0, 0)

        # Aggregate
        attention_exp = attention.unsqueeze(-1)  # (batch, max_neighbors, 1)
        kg_emb = torch.sum(attention_exp * tail_emb, dim=1)  # (batch, dim)

        return kg_emb

    def get_all_embeddings(self, device):
        """
        Pre-compute all user and item embeddings with GCN propagation.

        Returns:
            all_user_embed: (n_users, embedding_dim)
            all_item_embed: (n_items, embedding_dim)
        """
        # Get base embeddings
        user_embed_0 = self.user_embed.weight[1:]  # Skip index 0
        item_embed_0 = self.item_embed.weight[1:]

        # Get visual embeddings for all items
        visual_feat = self.visual_features.to(device)
        visual_emb = self.visual_proj(visual_feat)

        # Get KG embeddings for all items
        all_item_ids = torch.arange(1, self.n_items + 1, device=device)
        kg_emb = self.get_kg_enhanced_item_embed(all_item_ids, device)

        # Fuse: CF + Visual + KG
        concat = torch.cat([item_embed_0, visual_emb, kg_emb], dim=-1)
        gate = self.fusion_gate(concat)
        item_fused = gate * self.fusion_proj(concat) + (1 - gate) * item_embed_0

        # Combine for GCN
        all_embed = torch.cat([user_embed_0, item_fused], dim=0)

        # GCN propagation
        embed_layers = [all_embed]
        for _ in range(self.n_layers):
            all_embed = torch.sparse.mm(self.adj_matrix, all_embed)
            embed_layers.append(all_embed)

        # Mean pooling
        all_embed = torch.mean(torch.stack(embed_layers), dim=0)

        user_embed_final = all_embed[:self.n_users]
        item_embed_final = all_embed[self.n_users:]

        return user_embed_final, item_embed_final

    def forward(self, user_ids, item_ids):
        """
        Forward pass.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs

        Returns:
            scores: (batch_size,) predicted scores
        """
        device = user_ids.device

        # Get all embeddings with GCN
        all_user_emb, all_item_emb = self.get_all_embeddings(device)

        # Index for batch
        user_emb = all_user_emb[user_ids - 1]
        item_emb = all_item_emb[item_ids - 1]

        # Score
        scores = torch.sum(user_emb * item_emb, dim=-1)
        return scores

    def predict(self, user_ids, item_ids):
        """Predict scores (alias for forward)."""
        return self.forward(user_ids, item_ids)

    def get_reg_loss(self, user_ids, item_ids):
        """Compute L2 regularization loss."""
        user_emb = self.user_embed(user_ids)
        item_emb = self.item_embed(item_ids)

        reg_loss = (torch.norm(user_emb) ** 2 + torch.norm(item_emb) ** 2) / user_ids.shape[0]
        return self.reg_weight * reg_loss

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """
        Compute BPR loss.

        Args:
            user_ids: (batch_size,) user IDs
            pos_item_ids: (batch_size,) positive item IDs
            neg_item_ids: (batch_size,) negative item IDs

        Returns:
            loss: Total loss
            bpr_loss: BPR loss
            reg_loss: Regularization loss
        """
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        reg_loss = self.get_reg_loss(user_ids, pos_item_ids)
        reg_loss += self.get_reg_loss(user_ids, neg_item_ids)

        total_loss = bpr_loss + reg_loss

        return total_loss, bpr_loss, reg_loss
