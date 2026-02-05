#!/usr/bin/env python3
"""
Knowledge-Enhanced Heterogeneous Graph Recommendation Model

Full model: integrates CF view, KG view, Mask mechanism, and fusion layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional
import logging

from .encoders import CFEncoder, KGEncoder
from .losses import RecommendationLoss

logger = logging.getLogger(__name__)


class KnowledgeEnhancedRecModel(nn.Module):
    """Knowledge-Enhanced Recommendation Model (Ours-Full)"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_entities: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2,
        use_mask: bool = True,
        mask_init: Optional[torch.Tensor] = None,
        use_cf_view: bool = True,
        use_kg_view: bool = True,
    ):
        """
        Args:
            num_users: Number of users
            num_items: Number of items
            num_entities: Number of entities
            embedding_dim: Embedding dimension
            num_layers: Number of GNN layers
            gat_heads: GAT attention heads
            dropout: Dropout probability
            use_mask: Whether to use learnable Mask
            mask_init: Mask initial values (frequency-based)
            use_cf_view: Whether to use CF view
            use_kg_view: Whether to use KG view
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim

        self.use_mask = use_mask
        self.use_cf_view = use_cf_view
        self.use_kg_view = use_kg_view

        # === Embeddings ===
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.entity_embed = nn.Embedding(num_entities, embedding_dim)

        # Initialization
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)

        # === Learnable Mask ===
        if use_mask:
            if mask_init is not None:
                # Frequency-based initialization
                # Note: clamp to avoid logit(0)=-inf or logit(1)=inf
                eps = 1e-7
                mask_init_clamped = torch.clamp(mask_init, eps, 1 - eps)
                self.mask_logits = nn.Parameter(torch.logit(mask_init_clamped))
            else:
                # All-ones initialization (equivalent to no mask)
                self.mask_logits = nn.Parameter(torch.zeros(num_entities))
        else:
            self.register_buffer('mask_logits', torch.zeros(num_entities))

        # === CF View Encoder ===
        if use_cf_view:
            self.cf_encoder = CFEncoder(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                heads=gat_heads,
                dropout=dropout
            )
        else:
            self.cf_encoder = None

        # === KG View Encoder ===
        if use_kg_view:
            self.kg_encoder = KGEncoder(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                heads=gat_heads,
                dropout=dropout
            )
        else:
            self.kg_encoder = None

        # === Fusion Layer ===
        if use_cf_view and use_kg_view:
            # Both views enabled: fuse
            self.fusion_user = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim)
            )

            self.fusion_item = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            # Only one view: no fusion needed
            self.fusion_user = None
            self.fusion_item = None

    def get_mask(self) -> torch.Tensor:
        """Get current Mask values (sigmoid)"""
        if self.use_mask:
            return torch.sigmoid(self.mask_logits)
        else:
            return torch.ones(self.num_entities, device=self.mask_logits.device)

    def forward(
        self,
        hetero_graph: HeteroData,
        cf_edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            hetero_graph: Heterogeneous graph (contains User-Entity-Item edges)
            cf_edge_index: CF graph edge index (User-Item bidirectional edges)

        Returns:
            outputs: {
                'user_fused': [num_users, dim],   # for recommendation
                'item_fused': [num_items, dim],   # for recommendation
                'user_cf': [num_users, dim],       # Multi-view contrastive用
                'user_kg': [num_users, dim],       # Multi-view contrastive用
                'entity_emb': [num_entities, dim], # for alignment loss
                'item_kg': [num_items, dim],       # for alignment loss
                'mask': [num_entities],            # Mask regularization用
            }
        """
        outputs = {}

        # === 1. CF view encoding ===
        if self.use_cf_view:
            # Concatenate user and item embeddings
            x_cf = torch.cat([
                self.user_embed.weight,
                self.item_embed.weight
            ], dim=0)  # [num_users + num_items, dim]

            user_emb_cf, item_emb_cf = self.cf_encoder(x_cf, cf_edge_index)
            outputs['user_cf'] = user_emb_cf
            outputs['item_cf'] = item_emb_cf
        else:
            outputs['user_cf'] = None
            outputs['item_cf'] = None

        # === 2. KG view encoding (with mask)===
        if self.use_kg_view:
            # Apply mask to entity embeddings
            mask = self.get_mask()
            entity_emb_masked = self.entity_embed.weight * mask.unsqueeze(1)

            x_dict = {
                'user': self.user_embed.weight,
                'entity': entity_emb_masked,
                'item': self.item_embed.weight
            }

            # Heterogeneous GNN
            h_dict = self.kg_encoder(x_dict, hetero_graph.edge_index_dict)

            user_emb_kg = h_dict['user']
            entity_emb = h_dict['entity']
            item_emb_kg = h_dict['item']

            outputs['user_kg'] = user_emb_kg
            outputs['entity_emb'] = entity_emb
            outputs['item_kg'] = item_emb_kg
            outputs['mask'] = mask
        else:
            outputs['user_kg'] = None
            outputs['entity_emb'] = None
            outputs['item_kg'] = None
            outputs['mask'] = None

        # === 3. Fusion ===
        if self.use_cf_view and self.use_kg_view:
            # 两个视图都有：Fusion
            user_emb_fused = self.fusion_user(
                torch.cat([user_emb_cf, user_emb_kg], dim=-1)
            )
            item_emb_fused = self.fusion_item(
                torch.cat([item_emb_cf, item_emb_kg], dim=-1)
            )
        elif self.use_cf_view:
            # CF view only
            user_emb_fused = user_emb_cf
            item_emb_fused = item_emb_cf
        elif self.use_kg_view:
            # KG view only
            user_emb_fused = user_emb_kg
            item_emb_fused = item_emb_kg
        else:
            raise ValueError("At least one view (CF or KG) must be enabled")

        outputs['user_fused'] = user_emb_fused
        outputs['item_fused'] = item_emb_fused

        return outputs

    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict user preference scores for items

        Args:
            user_ids: [batch_size]
            item_ids: [batch_size] or [batch_size, num_items]
            user_emb: [num_users, dim]
            item_emb: [num_items, dim]

        Returns:
            scores: [batch_size] or [batch_size, num_items]
        """
        u_emb = user_emb[user_ids]  # [batch_size, dim]

        if item_ids.dim() == 1:
            # Single item per user
            i_emb = item_emb[item_ids]  # [batch_size, dim]
            scores = (u_emb * i_emb).sum(dim=-1)  # [batch_size]
        else:
            # Multiple items per user
            i_emb = item_emb[item_ids]  # [batch_size, num_items, dim]
            scores = (u_emb.unsqueeze(1) * i_emb).sum(dim=-1)  # [batch_size, num_items]

        return scores


if __name__ == '__main__':
    # Test model
    torch.manual_seed(42)

    num_users = 100
    num_items = 50
    num_entities = 30
    dim = 64

    # Create mock graph
    from torch_geometric.data import HeteroData

    hetero_graph = HeteroData()
    hetero_graph['user'].num_nodes = num_users
    hetero_graph['entity'].num_nodes = num_entities
    hetero_graph['item'].num_nodes = num_items

    hetero_graph['user', 'long_term', 'entity'].edge_index = torch.randint(
        0, min(num_users, num_entities), (2, 200)
    )
    hetero_graph['user', 'short_term', 'entity'].edge_index = torch.randint(
        0, min(num_users, num_entities), (2, 300)
    )
    hetero_graph['entity', 'describes', 'item'].edge_index = torch.randint(
        0, min(num_entities, num_items), (2, 150)
    )

    cf_edge_index = torch.randint(0, num_users + num_items, (2, 1000))

    # Create model
    model = KnowledgeEnhancedRecModel(
        num_users=num_users,
        num_items=num_items,
        num_entities=num_entities,
        embedding_dim=dim
    )

    # Forward pass
    outputs = model(hetero_graph, cf_edge_index)

    print("✓ Model outputs:")
    for key, value in outputs.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")

    # Test prediction
    user_ids = torch.randint(0, num_users, (10,))
    item_ids = torch.randint(0, num_items, (10,))
    scores = model.predict(user_ids, item_ids, outputs['user_fused'], outputs['item_fused'])
    print(f"\n✓ Prediction scores: {scores.shape}")
