#!/usr/bin/env python3
"""
MMGCN: Multi-modal Graph Convolution Network

Implementation based on:
"MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video" (ACM MM 2019)
by Wei et al.

Core idea: Build modality-specific user-item graphs and use GCN for propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix


class ModalityGCN(nn.Module):
    """
    GCN layer for a single modality.

    Propagates embeddings through user-item bipartite graph.
    """

    def __init__(self, embedding_dim, n_layers=2):
        """
        Args:
            embedding_dim: Embedding dimension
            n_layers: Number of GCN layers
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

    def forward(self, user_embed, item_embed, adj_matrix):
        """
        Propagate embeddings through graph.

        Args:
            user_embed: (n_users, embedding_dim) user embeddings
            item_embed: (n_items, embedding_dim) item embeddings
            adj_matrix: Sparse normalized adjacency matrix

        Returns:
            user_embed_layers: List of user embeddings from each layer
            item_embed_layers: List of item embeddings from each layer
        """
        # Combine user and item embeddings
        all_embed = torch.cat([user_embed, item_embed], dim=0)  # (n_users + n_items, embedding_dim)

        # Store embeddings from each layer
        embed_layers = [all_embed]

        # GCN propagation
        for layer in range(self.n_layers):
            # Message passing: multiply by adjacency matrix
            all_embed = torch.sparse.mm(adj_matrix, all_embed)
            embed_layers.append(all_embed)

        # Split back into user and item embeddings
        n_users = user_embed.shape[0]
        user_embed_layers = [emb[:n_users] for emb in embed_layers]
        item_embed_layers = [emb[n_users:] for emb in embed_layers]

        return user_embed_layers, item_embed_layers


class ModalAttention(nn.Module):
    """
    Modal attention mechanism.

    Learns importance weights for different modalities.
    """

    def __init__(self, embedding_dim, n_modalities=2):
        """
        Args:
            embedding_dim: Embedding dimension
            n_modalities: Number of modalities
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_modalities = n_modalities

        # Attention weights for each modality
        self.attention_weights = nn.Parameter(torch.ones(n_modalities))

    def forward(self, modality_embeds):
        """
        Compute attention-weighted fusion of modality embeddings.

        Args:
            modality_embeds: List of (batch_size, embedding_dim) embeddings for each modality

        Returns:
            fused_embed: (batch_size, embedding_dim) attention-weighted embedding
        """
        # Softmax over modalities
        attention = F.softmax(self.attention_weights, dim=0)

        # Weighted sum
        fused_embed = sum(attention[i] * emb for i, emb in enumerate(modality_embeds))

        return fused_embed, attention


class MMGCN(nn.Module):
    """
    Multi-modal Graph Convolution Network.

    Integrates visual features with collaborative filtering using modality-specific GCNs.
    """

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        visual_dim=2048,
        n_layers=2,
        reg_weight=1e-5
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            visual_dim: Dimension of visual features (ResNet50: 2048)
            n_layers: Number of GCN layers
            reg_weight: L2 regularization weight
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.visual_dim = visual_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        # CF modality: Standard user/item embeddings (RecBole 1-indexed)
        self.user_embed_cf = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embed_cf = nn.Embedding(n_items + 1, embedding_dim)

        # Visual modality: User embeddings + visual projection for items
        self.user_embed_visual = nn.Embedding(n_users + 1, embedding_dim)

        # Visual feature projection
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # GCN for each modality
        self.gcn_cf = ModalityGCN(embedding_dim, n_layers)
        self.gcn_visual = ModalityGCN(embedding_dim, n_layers)

        # Modal attention
        self.modal_attention = ModalAttention(embedding_dim, n_modalities=2)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embed_cf.weight)
        nn.init.xavier_uniform_(self.item_embed_cf.weight)
        nn.init.xavier_uniform_(self.user_embed_visual.weight)

        # Adjacency matrix placeholder
        self.adj_matrix = None
        self.visual_features = None

    def load_visual_features(self, visual_features):
        """
        Load precomputed visual features.

        Args:
            visual_features: numpy array of shape (n_items, visual_dim)
        """
        self.visual_features = torch.FloatTensor(visual_features)

    def build_adjacency_matrix(self, train_interactions, device):
        """
        Build normalized adjacency matrix for user-item bipartite graph.

        Args:
            train_interactions: List of (user, item, rating) tuples
            device: torch device

        Returns:
            Normalized sparse adjacency matrix
        """
        print("Building adjacency matrix...")

        # Build interaction matrix
        row_user = []
        col_item = []

        for u, i, r in train_interactions:
            row_user.append(u - 1)  # 0-indexed
            col_item.append(i - 1)

        # Create bipartite graph adjacency matrix
        # A = [[0, R],
        #      [R^T, 0]]
        # Where R is user-item interaction matrix

        n_interactions = len(row_user)
        data = np.ones(n_interactions)

        # User-Item block
        R = coo_matrix((data, (row_user, col_item)), shape=(self.n_users, self.n_items))

        # Adjacency matrix (symmetric)
        # Top: [0, R]
        # Bottom: [R^T, 0]
        from scipy.sparse import vstack, hstack

        top = hstack([coo_matrix((self.n_users, self.n_users)), R])
        bottom = hstack([R.T, coo_matrix((self.n_items, self.n_items))])
        A = vstack([top, bottom])

        # Normalize: D^(-1/2) * A * D^(-1/2)
        # Compute degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv_sqrt = np.power(degrees, -0.5)
        D_inv_sqrt = coo_matrix((D_inv_sqrt, (np.arange(len(D_inv_sqrt)), np.arange(len(D_inv_sqrt)))))

        # Normalized adjacency
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        # Convert to PyTorch sparse tensor
        A_norm_coo = A_norm.tocoo()
        indices = torch.LongTensor([A_norm_coo.row, A_norm_coo.col])
        values = torch.FloatTensor(A_norm_coo.data)
        shape = A_norm_coo.shape

        adj_matrix = torch.sparse.FloatTensor(indices, values, torch.Size(shape)).to(device)

        print(f"✓ Adjacency matrix built: {shape}")
        return adj_matrix

    def forward_modality(self, user_ids, item_ids, modality='cf'):
        """
        Forward pass for a single modality.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs
            modality: 'cf' or 'visual'

        Returns:
            user_embed: (batch_size, embedding_dim)
            item_embed: (batch_size, embedding_dim)
        """
        device = user_ids.device

        if modality == 'cf':
            # CF modality: standard embeddings
            user_embed_0 = self.user_embed_cf.weight[1:]  # Skip index 0
            item_embed_0 = self.item_embed_cf.weight[1:]

            # GCN propagation
            user_layers, item_layers = self.gcn_cf(user_embed_0, item_embed_0, self.adj_matrix)

            # Combine layers (mean pooling)
            user_embed_all = torch.mean(torch.stack(user_layers), dim=0)
            item_embed_all = torch.mean(torch.stack(item_layers), dim=0)

        elif modality == 'visual':
            # Visual modality: user embeddings + projected visual features
            user_embed_0 = self.user_embed_visual.weight[1:]  # Skip index 0

            # Project visual features for all items
            visual_feat = self.visual_features.to(device)
            item_embed_0 = self.visual_proj(visual_feat)

            # GCN propagation
            user_layers, item_layers = self.gcn_visual(user_embed_0, item_embed_0, self.adj_matrix)

            # Combine layers
            user_embed_all = torch.mean(torch.stack(user_layers), dim=0)
            item_embed_all = torch.mean(torch.stack(item_layers), dim=0)

        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Index embeddings for batch
        user_embed = user_embed_all[user_ids - 1]  # 0-indexed
        item_embed = item_embed_all[item_ids - 1]

        return user_embed, item_embed

    def forward(self, user_ids, item_ids):
        """
        Forward pass with modal fusion.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs

        Returns:
            scores: (batch_size,) predicted scores
        """
        # Get embeddings from each modality
        user_embed_cf, item_embed_cf = self.forward_modality(user_ids, item_ids, 'cf')
        user_embed_visual, item_embed_visual = self.forward_modality(user_ids, item_ids, 'visual')

        # Fuse with attention
        user_embed, _ = self.modal_attention([user_embed_cf, user_embed_visual])
        item_embed, _ = self.modal_attention([item_embed_cf, item_embed_visual])

        # Compute scores
        scores = torch.sum(user_embed * item_embed, dim=-1)

        return scores

    def predict(self, user_ids, item_ids):
        """Predict scores (alias for forward)."""
        return self.forward(user_ids, item_ids)

    def get_all_embeddings(self, device):
        """
        Pre-compute all user and item embeddings with GCN propagation.

        Returns:
            all_user_embed: (n_users, embedding_dim)
            all_item_embed: (n_items, embedding_dim)
        """
        # CF modality
        user_embed_0_cf = self.user_embed_cf.weight[1:].to(device)
        item_embed_0_cf = self.item_embed_cf.weight[1:].to(device)
        user_layers_cf, item_layers_cf = self.gcn_cf(user_embed_0_cf, item_embed_0_cf, self.adj_matrix)
        user_embed_cf = torch.mean(torch.stack(user_layers_cf), dim=0)
        item_embed_cf = torch.mean(torch.stack(item_layers_cf), dim=0)

        # Visual modality
        user_embed_0_visual = self.user_embed_visual.weight[1:].to(device)
        visual_feat = self.visual_features.to(device)
        item_embed_0_visual = self.visual_proj(visual_feat)
        user_layers_visual, item_layers_visual = self.gcn_visual(user_embed_0_visual, item_embed_0_visual, self.adj_matrix)
        user_embed_visual = torch.mean(torch.stack(user_layers_visual), dim=0)
        item_embed_visual = torch.mean(torch.stack(item_layers_visual), dim=0)

        # Fuse with attention
        user_embed, _ = self.modal_attention([user_embed_cf, user_embed_visual])
        item_embed, _ = self.modal_attention([item_embed_cf, item_embed_visual])

        return user_embed, item_embed

    def get_reg_loss(self, user_ids, item_ids):
        """
        Compute L2 regularization loss.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs

        Returns:
            reg_loss: Regularization loss
        """
        # CF embeddings
        user_embed_cf = self.user_embed_cf(user_ids)
        item_embed_cf = self.item_embed_cf(item_ids)

        # Visual embeddings
        user_embed_visual = self.user_embed_visual(user_ids)

        # L2 norm
        reg_loss = (
            torch.norm(user_embed_cf) ** 2 +
            torch.norm(item_embed_cf) ** 2 +
            torch.norm(user_embed_visual) ** 2
        ) / user_ids.shape[0]

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
        # Predict scores
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)

        # BPR loss
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization
        reg_loss = self.get_reg_loss(user_ids, pos_item_ids)
        reg_loss += self.get_reg_loss(user_ids, neg_item_ids)

        total_loss = bpr_loss + reg_loss

        return total_loss, bpr_loss, reg_loss
