#!/usr/bin/env python3
"""
VBPR: Visual Bayesian Personalized Ranking

Implementation based on:
"VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback" (AAAI 2016)
by Ruining He and Julian McAuley

Core idea: Extend BPR with visual features from CNN.
Prediction: score(u,i) = u·i + u·(E·v_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VBPR(nn.Module):
    """
    Visual Bayesian Personalized Ranking.

    Combines standard matrix factorization with visual features.
    """

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        visual_dim=2048,
        reg_weight=1e-5
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of user/item embeddings
            visual_dim: Dimension of visual features (ResNet50: 2048)
            reg_weight: L2 regularization weight
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.visual_dim = visual_dim
        self.reg_weight = reg_weight

        # User and item embeddings (RecBole uses 1-indexed)
        self.user_embed = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embed = nn.Embedding(n_items + 1, embedding_dim)

        # Visual embedding matrix E: projects visual features to embedding space
        # E: (embedding_dim, visual_dim)
        # So E · v gives (embedding_dim,) vector
        self.visual_embed = nn.Linear(visual_dim, embedding_dim, bias=False)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)
        nn.init.xavier_uniform_(self.visual_embed.weight)

        # Visual features placeholder
        self.visual_features = None

    def load_visual_features(self, visual_features):
        """
        Load precomputed visual features.

        Args:
            visual_features: numpy array of shape (n_items, visual_dim)
        """
        self.visual_features = torch.FloatTensor(visual_features)

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

        # Get embeddings
        u_embed = self.user_embed(user_ids)  # (batch_size, embedding_dim)
        i_embed = self.item_embed(item_ids)  # (batch_size, embedding_dim)

        # CF component: u · i
        cf_score = torch.sum(u_embed * i_embed, dim=-1)  # (batch_size,)

        # Visual component: u · (E · v_i)
        # Get visual features for items (RecBole IDs are 1-indexed)
        visual_feat = self.visual_features[item_ids.cpu() - 1].to(device)  # (batch_size, visual_dim)

        # Project visual features: E · v_i
        visual_embed = self.visual_embed(visual_feat)  # (batch_size, embedding_dim)

        # User's affinity to visual features: u · (E · v_i)
        visual_score = torch.sum(u_embed * visual_embed, dim=-1)  # (batch_size,)

        # Total score
        scores = cf_score + visual_score

        return scores

    def predict(self, user_ids, item_ids):
        """
        Predict scores (alias for forward).

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs

        Returns:
            scores: (batch_size,) predicted scores
        """
        return self.forward(user_ids, item_ids)

    def get_reg_loss(self, user_ids, item_ids):
        """
        Compute L2 regularization loss.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs

        Returns:
            reg_loss: Regularization loss
        """
        u_embed = self.user_embed(user_ids)
        i_embed = self.item_embed(item_ids)

        # L2 norm of embeddings
        reg_loss = (torch.norm(u_embed) ** 2 + torch.norm(i_embed) ** 2) / user_ids.shape[0]

        # Also regularize visual embedding matrix
        # Note: visual_embed.weight is the E matrix
        reg_loss += torch.norm(self.visual_embed.weight) ** 2 / self.n_items

        return self.reg_weight * reg_loss

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """
        Compute BPR (Bayesian Personalized Ranking) loss.

        Args:
            user_ids: (batch_size,) user IDs
            pos_item_ids: (batch_size,) positive item IDs
            neg_item_ids: (batch_size,) negative item IDs

        Returns:
            loss: BPR loss + regularization
        """
        # Predict scores for positive and negative items
        pos_scores = self.forward(user_ids, pos_item_ids)  # (batch_size,)
        neg_scores = self.forward(user_ids, neg_item_ids)  # (batch_size,)

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization
        reg_loss = self.get_reg_loss(user_ids, pos_item_ids)

        # Add regularization for negative items
        reg_loss += self.get_reg_loss(user_ids, neg_item_ids)

        total_loss = bpr_loss + reg_loss

        return total_loss, bpr_loss, reg_loss
