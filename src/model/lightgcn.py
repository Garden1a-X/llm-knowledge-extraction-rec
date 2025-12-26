"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

Paper: He et al., LightGCN: Simplifying and Powering Graph Convolution Network
       for Recommendation, SIGIR 2020

Key ideas:
- Remove feature transformation and nonlinear activation
- Simple weighted sum aggregation of neighbor embeddings
- Layer combination by averaging all layer embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from typing import Optional, Tuple


class LightGCN(nn.Module):
    """
    LightGCN model for collaborative filtering.

    The model consists of:
    1. Embedding layer for users and items
    2. Multiple GCN layers (simplified, no transformation/activation)
    3. Layer combination (average pooling)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0,
        add_self_loops: bool = False
    ):
        """
        Initialize LightGCN.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embeddings
            n_layers: Number of GCN layers
            dropout: Dropout rate (0 means no dropout)
            add_self_loops: Whether to add self-loops to graph
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Cache for graph structure
        self.edge_index = None
        self.edge_weight = None

    def compute_graph_embeddings(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute user and item embeddings via graph convolution.

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Get initial embeddings (layer 0)
        # Shape: [n_users + n_items, embedding_dim]
        all_embeddings = torch.cat([
            self.user_embedding.weight,  # [n_users, embedding_dim]
            self.item_embedding.weight   # [n_items, embedding_dim]
        ], dim=0)

        # Store embeddings from all layers
        embeddings_list = [all_embeddings]

        # Graph convolution layers
        for layer in range(self.n_layers):
            # Simple message passing: aggregate neighbor embeddings
            all_embeddings = self._propagate(
                all_embeddings,
                edge_index,
                edge_weight
            )

            # Dropout
            if self.training and self.dropout > 0:
                all_embeddings = F.dropout(
                    all_embeddings,
                    p=self.dropout,
                    training=self.training
                )

            embeddings_list.append(all_embeddings)

        # Layer combination: average all layers
        # Shape: [n_users + n_items, embedding_dim]
        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        # Split into user and item embeddings
        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]

        return user_embeddings, item_embeddings

    def _propagate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagate embeddings through one GCN layer.

        Implements: H^(l+1) = D^(-1/2) A D^(-1/2) H^(l)

        Args:
            x: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Updated node embeddings [num_nodes, embedding_dim]
        """
        # Get edge indices
        row, col = edge_index[0], edge_index[1]

        # Compute normalization: 1 / sqrt(deg(i) * deg(j))
        if edge_weight is None:
            # Compute degree
            deg_row = degree(row, x.size(0), dtype=x.dtype)
            deg_col = degree(col, x.size(0), dtype=x.dtype)

            # Normalization: 1 / sqrt(deg(i) * deg(j))
            deg_row_inv_sqrt = deg_row.pow(-0.5)
            deg_col_inv_sqrt = deg_col.pow(-0.5)
            deg_row_inv_sqrt[deg_row_inv_sqrt == float('inf')] = 0
            deg_col_inv_sqrt[deg_col_inv_sqrt == float('inf')] = 0

            edge_weight = deg_row_inv_sqrt[row] * deg_col_inv_sqrt[col]

        # Message passing: aggregate neighbor embeddings
        # out[i] = sum_j (edge_weight[i,j] * x[j])
        out = torch.zeros_like(x)
        out.index_add_(0, row, edge_weight.view(-1, 1) * x[col])

        return out

    def forward(
        self,
        edge_index: torch.Tensor,
        users: Optional[torch.Tensor] = None,
        pos_items: Optional[torch.Tensor] = None,
        neg_items: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training with BPR loss.

        Args:
            edge_index: Graph edge indices [2, num_edges]
            users: User indices [batch_size]
            pos_items: Positive item indices [batch_size]
            neg_items: Negative item indices [batch_size]

        Returns:
            Tuple of (user_embeddings, pos_item_embeddings, neg_item_embeddings)
        """
        # Compute graph embeddings
        user_embeddings, item_embeddings = self.compute_graph_embeddings(edge_index)

        if users is not None and pos_items is not None and neg_items is not None:
            # Get embeddings for specific users and items
            batch_user_emb = user_embeddings[users]
            batch_pos_item_emb = item_embeddings[pos_items]
            batch_neg_item_emb = item_embeddings[neg_items]

            return batch_user_emb, batch_pos_item_emb, batch_neg_item_emb
        else:
            # Return all embeddings (for inference)
            return user_embeddings, item_embeddings, None

    def predict(
        self,
        edge_index: torch.Tensor,
        users: torch.Tensor,
        items: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Args:
            edge_index: Graph edge indices
            users: User indices [batch_size] or [1]
            items: Item indices [num_items] or None (all items)

        Returns:
            Prediction scores [batch_size, num_items] or [batch_size]
        """
        self.eval()
        with torch.no_grad():
            # Compute embeddings
            user_embeddings, item_embeddings = self.compute_graph_embeddings(edge_index)

            # Get user embeddings
            batch_user_emb = user_embeddings[users]  # [batch_size, dim]

            if items is None:
                # Predict for all items
                scores = torch.matmul(batch_user_emb, item_embeddings.t())  # [batch_size, n_items]
            else:
                # Predict for specific items
                batch_item_emb = item_embeddings[items]  # [num_items, dim]
                scores = torch.matmul(batch_user_emb, batch_item_emb.t())  # [batch_size, num_items]

                # If single user, return flat scores
                if users.size(0) == 1:
                    scores = scores.squeeze(0)  # [num_items]

            return scores

    def bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        reg_weight: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bayesian Personalized Ranking loss.

        BPR loss: -ln(sigmoid(pos_score - neg_score))

        Args:
            user_emb: User embeddings [batch_size, dim]
            pos_item_emb: Positive item embeddings [batch_size, dim]
            neg_item_emb: Negative item embeddings [batch_size, dim]
            reg_weight: L2 regularization weight

        Returns:
            Tuple of (total_loss, bpr_loss, reg_loss)
        """
        # Compute scores
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # [batch_size]
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)  # [batch_size]

        # BPR loss: -ln(sigmoid(pos - neg))
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 regularization
        reg_loss = reg_weight * (
            user_emb.norm(2).pow(2) +
            pos_item_emb.norm(2).pow(2) +
            neg_item_emb.norm(2).pow(2)
        ) / user_emb.size(0)

        total_loss = bpr_loss + reg_loss

        return total_loss, bpr_loss, reg_loss

    def get_embedding(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get final user and item embeddings.

        Args:
            edge_index: Graph edge indices

        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        self.eval()
        with torch.no_grad():
            return self.compute_graph_embeddings(edge_index)
