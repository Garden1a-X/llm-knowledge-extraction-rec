#!/usr/bin/env python3
"""
MKGAT: Multi-modal Knowledge Graph Attention Network

Implementation based on:
"Multi-modal Knowledge Graphs for Recommender Systems" (CIKM 2020)

Core components:
1. Visual features from ResNet50
2. Knowledge graph with attention-based aggregation
3. Multi-modal fusion for recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Aggregator(nn.Module):
    """
    Knowledge Graph Attention Aggregator.

    Aggregates neighbor information with attention mechanism,
    considering both entities and relations.
    """

    def __init__(self, in_dim, out_dim, dropout=0.1, aggregator_type='bi-interaction'):
        """
        Args:
            in_dim: Input embedding dimension
            out_dim: Output embedding dimension
            dropout: Dropout rate
            aggregator_type: 'bi-interaction', 'gcn', or 'graphsage'
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator_type = aggregator_type

        # Transformation matrices
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        if aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(in_dim, out_dim, bias=False)
            self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        elif aggregator_type == 'graphsage':
            self.W_concat = nn.Linear(in_dim * 2, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, ego_embed, neighbor_embed, relation_embed):
        """
        Aggregate neighbor information with attention.

        Args:
            ego_embed: (batch_size, in_dim) - center entity embeddings
            neighbor_embed: (batch_size, n_neighbors, in_dim) - neighbor embeddings
            relation_embed: (batch_size, n_neighbors, in_dim) - relation embeddings

        Returns:
            Aggregated embeddings (batch_size, out_dim)
        """
        # Attention scores: consider both entity and relation
        # score = <ego, relation * neighbor>
        neighbor_relation = neighbor_embed * relation_embed  # Element-wise

        # Compute attention scores
        ego_expanded = ego_embed.unsqueeze(1)  # (batch_size, 1, in_dim)
        scores = torch.sum(ego_expanded * neighbor_relation, dim=-1)  # (batch_size, n_neighbors)

        # Attention weights
        attention = F.softmax(scores, dim=-1)  # (batch_size, n_neighbors)
        attention = self.dropout(attention)

        # Weighted aggregation
        attention_expanded = attention.unsqueeze(-1)  # (batch_size, n_neighbors, 1)
        neighbor_agg = torch.sum(attention_expanded * neighbor_embed, dim=1)  # (batch_size, in_dim)

        # Different aggregation strategies
        if self.aggregator_type == 'gcn':
            # GCN-style: mean aggregation
            output = self.W(ego_embed + neighbor_agg)

        elif self.aggregator_type == 'graphsage':
            # GraphSAGE-style: concat then transform
            concat = torch.cat([ego_embed, neighbor_agg], dim=-1)
            output = self.W_concat(concat)

        elif self.aggregator_type == 'bi-interaction':
            # Bi-Interaction: element-wise + feature-wise
            sum_embed = self.W1(ego_embed + neighbor_agg)
            bi_embed = self.W2(ego_embed * neighbor_agg)
            output = sum_embed + bi_embed

        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")

        return F.leaky_relu(output)


class MKGAT(nn.Module):
    """
    Multi-modal Knowledge Graph Attention Network.

    Integrates visual features from ResNet50 with knowledge graph embeddings
    using attention-based aggregation.
    """

    def __init__(
        self,
        n_users,
        n_items,
        n_entities,
        n_relations,
        embedding_dim=64,
        visual_dim=2048,
        n_layers=2,
        aggregator_type='bi-interaction',
        dropout=0.1,
        reg_weight=1e-5
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            n_entities: Number of entities in KG
            n_relations: Number of relations in KG
            embedding_dim: Embedding dimension
            visual_dim: Dimension of visual features (ResNet50: 2048)
            n_layers: Number of aggregation layers
            aggregator_type: Type of aggregator ('bi-interaction', 'gcn', 'graphsage')
            dropout: Dropout rate
            reg_weight: L2 regularization weight
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.visual_dim = visual_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        # User and entity embeddings
        # RecBole uses 1-indexed IDs, so we need n+1 embeddings (index 0 unused)
        self.user_embed = nn.Embedding(n_users + 1, embedding_dim)
        self.entity_embed = nn.Embedding(n_entities + 1, embedding_dim)
        self.relation_embed = nn.Embedding(n_relations + 1, embedding_dim)

        # Visual feature projection
        # Project 2048-dim ResNet features to embedding_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Multi-modal fusion for items
        # Fuse visual features with entity embeddings
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # KG aggregation layers
        self.aggregators = nn.ModuleList()
        for _ in range(n_layers):
            self.aggregators.append(
                Aggregator(embedding_dim, embedding_dim, dropout, aggregator_type)
            )

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    def load_visual_features(self, visual_features):
        """
        Load precomputed visual features.

        Args:
            visual_features: numpy array of shape (n_items, 2048)
        """
        self.visual_features = torch.FloatTensor(visual_features)

    def get_item_embeddings(self, item_ids, device):
        """
        Get multimodal item embeddings (visual + KG entity).

        Args:
            item_ids: (batch_size,) item IDs
            device: torch device

        Returns:
            Item embeddings (batch_size, embedding_dim)
        """
        # Get entity embeddings for items
        # Assuming item IDs map directly to entity IDs (1-indexed)
        entity_ids = item_ids

        entity_emb = self.entity_embed(entity_ids)

        # Get visual features
        # Move item_ids to CPU for indexing, then move result to device
        # RecBole IDs are 1-indexed, but numpy arrays are 0-indexed
        visual_feat = self.visual_features[item_ids.cpu() - 1].to(device)
        visual_emb = self.visual_proj(visual_feat)

        # Fuse visual and entity embeddings
        multimodal_emb = torch.cat([visual_emb, entity_emb], dim=-1)
        item_emb = self.fusion(multimodal_emb)

        return item_emb

    def aggregate_neighbors(self, entity_ids, adj_entity, adj_relation, layer_idx):
        """
        Aggregate neighbor information for given entities.

        Args:
            entity_ids: (batch_size,) entity IDs
            adj_entity: (batch_size, n_neighbors) neighbor entity IDs
            adj_relation: (batch_size, n_neighbors) relation IDs
            layer_idx: Which aggregation layer to use

        Returns:
            Aggregated embeddings (batch_size, embedding_dim)
        """
        # Get embeddings
        ego_embed = self.entity_embed(entity_ids)
        neighbor_embed = self.entity_embed(adj_entity)
        relation_embed = self.relation_embed(adj_relation)

        # Aggregate
        aggregator = self.aggregators[layer_idx]
        agg_embed = aggregator(ego_embed, neighbor_embed, relation_embed)

        return agg_embed

    def forward(self, user_ids, item_ids, adj_entity, adj_relation):
        """
        Forward pass.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs
            adj_entity: (batch_size, n_layers, n_neighbors) neighbor entities
            adj_relation: (batch_size, n_layers, n_neighbors) relations

        Returns:
            user_embed: (batch_size, embedding_dim)
            item_embed: (batch_size, embedding_dim)
        """
        device = user_ids.device

        # User embeddings
        user_embed = self.user_embed(user_ids)

        # Item embeddings with visual features
        item_embed = self.get_item_embeddings(item_ids, device)

        # Multi-hop aggregation for items through KG
        entity_ids = item_ids  # Items are entities in the KG
        entity_layers = [item_embed]

        for layer in range(self.n_layers):
            # Get neighbors for this layer
            layer_adj_entity = adj_entity[:, layer, :]
            layer_adj_relation = adj_relation[:, layer, :]

            # Aggregate
            agg_embed = self.aggregate_neighbors(
                entity_ids, layer_adj_entity, layer_adj_relation, layer
            )

            entity_layers.append(agg_embed)

        # Combine all layers (mean pooling)
        item_embed_final = torch.mean(torch.stack(entity_layers), dim=0)

        return user_embed, item_embed_final

    def predict(self, user_ids, item_ids, adj_entity, adj_relation):
        """
        Predict user-item scores.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs
            adj_entity: (batch_size, n_layers, n_neighbors) neighbor entities
            adj_relation: (batch_size, n_layers, n_neighbors) relations

        Returns:
            Predicted scores (batch_size,)
        """
        user_embed, item_embed = self.forward(user_ids, item_ids, adj_entity, adj_relation)

        # Dot product for scoring
        scores = torch.sum(user_embed * item_embed, dim=-1)

        return scores

    def get_reg_loss(self, user_ids, item_ids):
        """
        Compute L2 regularization loss.

        Args:
            user_ids: (batch_size,) user IDs
            item_ids: (batch_size,) item IDs

        Returns:
            Regularization loss
        """
        user_embed = self.user_embed(user_ids)
        entity_embed = self.entity_embed(item_ids)

        reg_loss = torch.norm(user_embed) ** 2 + torch.norm(entity_embed) ** 2
        reg_loss = reg_loss / user_ids.shape[0]

        return self.reg_weight * reg_loss
