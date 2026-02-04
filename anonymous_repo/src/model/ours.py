#!/usr/bin/env python3
"""
Knowledge-Enhanced Heterogeneous Graph Recommendation Model

完整模型：集成CF视图、KG视图、Mask机制和融合层。
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
    """知识增强推荐模型（Ours-Full）"""

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
            num_users: 用户数量
            num_items: 物品数量
            num_entities: Entity数量
            embedding_dim: Embedding维度
            num_layers: GNN层数
            gat_heads: GAT attention heads
            dropout: Dropout概率
            use_mask: 是否使用可学习Mask
            mask_init: Mask初始值（基于频率）
            use_cf_view: 是否使用CF视图
            use_kg_view: 是否使用KG视图
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

        # 初始化
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)

        # === 可学习Mask ===
        if use_mask:
            if mask_init is not None:
                # 基于频率初始化
                # 注意：需要clamp避免logit(0)=-inf或logit(1)=inf
                eps = 1e-7
                mask_init_clamped = torch.clamp(mask_init, eps, 1 - eps)
                self.mask_logits = nn.Parameter(torch.logit(mask_init_clamped))
            else:
                # 全1初始化（等价于no mask）
                self.mask_logits = nn.Parameter(torch.zeros(num_entities))
        else:
            self.register_buffer('mask_logits', torch.zeros(num_entities))

        # === CF视图编码器 ===
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

        # === KG视图编码器 ===
        if use_kg_view:
            self.kg_encoder = KGEncoder(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                heads=gat_heads,
                dropout=dropout
            )
        else:
            self.kg_encoder = None

        # === 融合层 ===
        if use_cf_view and use_kg_view:
            # 两个视图都用：融合
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
            # 只用一个视图：不需要融合
            self.fusion_user = None
            self.fusion_item = None

    def get_mask(self) -> torch.Tensor:
        """获取当前Mask值（sigmoid）"""
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
            hetero_graph: 异构图（包含User-Entity-Item边）
            cf_edge_index: CF图边索引（User-Item双向边）

        Returns:
            outputs: {
                'user_fused': [num_users, dim],   # 推荐用
                'item_fused': [num_items, dim],   # 推荐用
                'user_cf': [num_users, dim],       # 多视图对比用
                'user_kg': [num_users, dim],       # 多视图对比用
                'entity_emb': [num_entities, dim], # 对齐损失用
                'item_kg': [num_items, dim],       # 对齐损失用
                'mask': [num_entities],            # Mask正则用
            }
        """
        outputs = {}

        # === 1. CF视图编码 ===
        if self.use_cf_view:
            # 拼接User和Item embeddings
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

        # === 2. KG视图编码（带Mask）===
        if self.use_kg_view:
            # 应用Mask到Entity embedding
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

        # === 3. 融合 ===
        if self.use_cf_view and self.use_kg_view:
            # 两个视图都有：融合
            user_emb_fused = self.fusion_user(
                torch.cat([user_emb_cf, user_emb_kg], dim=-1)
            )
            item_emb_fused = self.fusion_item(
                torch.cat([item_emb_cf, item_emb_kg], dim=-1)
            )
        elif self.use_cf_view:
            # 只用CF视图
            user_emb_fused = user_emb_cf
            item_emb_fused = item_emb_cf
        elif self.use_kg_view:
            # 只用KG视图
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
        预测用户对物品的偏好分数

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
    # 测试模型
    torch.manual_seed(42)

    num_users = 100
    num_items = 50
    num_entities = 30
    dim = 64

    # 创建模拟图
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

    # 创建模型
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

    # 测试预测
    user_ids = torch.randint(0, num_users, (10,))
    item_ids = torch.randint(0, num_items, (10,))
    scores = model.predict(user_ids, item_ids, outputs['user_fused'], outputs['item_fused'])
    print(f"\n✓ Prediction scores: {scores.shape}")
