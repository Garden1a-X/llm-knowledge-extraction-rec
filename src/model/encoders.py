#!/usr/bin/env python3
"""
Graph Encoders: CF View and KG View

CF编码器：User-Item二部图（GAT）
KG编码器：User-Entity-Item异构图（HeteroConv + GAT）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CFEncoder(nn.Module):
    """
    CF视图编码器（传统协同过滤图）

    结构：User-Item二部图，使用GAT进行消息传递
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            num_users: 用户数量
            num_items: 物品数量
            embedding_dim: Embedding维度
            num_layers: GNN层数
            heads: GAT attention heads
            dropout: Dropout概率
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User和Item的初始embedding（在完整模型中从外部传入）
        # 这里只定义GNN layers

        # GAT layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer
                in_channels = embedding_dim
                out_channels = embedding_dim
                concat = True
            elif i < num_layers - 1:
                # Middle layers
                in_channels = embedding_dim * heads
                out_channels = embedding_dim
                concat = True
            else:
                # Last layer
                in_channels = embedding_dim * heads
                out_channels = embedding_dim
                concat = False  # No concat in final layer

            self.convs.append(
                GATConv(
                    in_channels,
                    out_channels,
                    heads=heads if concat else 1,
                    dropout=dropout,
                    concat=concat
                )
            )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [num_users + num_items, dim] - 拼接的User和Item embeddings
            edge_index: [2, num_edges] - User-Item边（双向）

        Returns:
            user_emb: [num_users, dim]
            item_emb: [num_items, dim]
        """
        h = x

        # Graph convolution
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Split into user and item embeddings
        user_emb = h[:self.num_users]
        item_emb = h[self.num_users:]

        return user_emb, item_emb


class KGEncoder(nn.Module):
    """
    KG视图编码器（知识增强异构图）

    结构：User-Entity-Item异构图，使用HeteroConv + GAT
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            embedding_dim: Embedding维度
            num_layers: GNN层数
            heads: GAT attention heads
            dropout: Dropout概率
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer
                in_channels = embedding_dim
                out_channels = embedding_dim
                concat = True
            elif i < num_layers - 1:
                # Middle layers
                in_channels = embedding_dim * heads
                out_channels = embedding_dim
                concat = True
            else:
                # Last layer
                in_channels = embedding_dim * heads
                out_channels = embedding_dim
                concat = False

            # HeteroConv包含多种边类型的GATConv
            # 注意：异构边不能添加self-loops
            hetero_conv = HeteroConv({
                ('user', 'long_term', 'entity'): GATConv(
                    in_channels,
                    out_channels,
                    heads=heads if concat else 1,
                    dropout=dropout,
                    concat=concat,
                    add_self_loops=False  # 异构边不能添加自环
                ),
                ('user', 'short_term', 'entity'): GATConv(
                    in_channels,
                    out_channels,
                    heads=heads if concat else 1,
                    dropout=dropout,
                    concat=concat,
                    add_self_loops=False
                ),
                ('entity', 'describes', 'item'): GATConv(
                    in_channels,
                    out_channels,
                    heads=heads if concat else 1,
                    dropout=dropout,
                    concat=concat,
                    add_self_loops=False
                ),
            }, aggr='sum')  # 不同边类型的聚合方式

            self.convs.append(hetero_conv)

        self.dropout = dropout

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: {
                'user': [num_users, dim],
                'entity': [num_entities, dim],
                'item': [num_items, dim]
            }
            edge_index_dict: {
                ('user', 'long_term', 'entity'): edge_index,
                ('user', 'short_term', 'entity'): edge_index,
                ('entity', 'describes', 'item'): edge_index,
            }

        Returns:
            out_dict: {
                'user': [num_users, dim],
                'entity': [num_entities, dim],
                'item': [num_items, dim]
            }
        """
        h_dict = x_dict

        # Heterogeneous graph convolution
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)

            if i < len(self.convs) - 1:
                # ReLU和Dropout（除了最后一层）
                h_dict = {
                    key: F.dropout(F.relu(h), p=self.dropout, training=self.training)
                    for key, h in h_dict.items()
                }

        return h_dict


if __name__ == '__main__':
    # 测试编码器
    torch.manual_seed(42)

    num_users = 100
    num_items = 50
    num_entities = 30
    dim = 64

    # === 测试CF编码器 ===
    print("Testing CF Encoder...")

    # 创建CF图
    num_cf_edges = 500
    cf_edge_index = torch.randint(0, num_users + num_items, (2, num_cf_edges * 2))

    # 初始embeddings
    cf_x = torch.randn(num_users + num_items, dim)

    cf_encoder = CFEncoder(num_users, num_items, dim, num_layers=2)
    user_emb_cf, item_emb_cf = cf_encoder(cf_x, cf_edge_index)

    print(f"  User embedding: {user_emb_cf.shape}")
    print(f"  Item embedding: {item_emb_cf.shape}")

    # === 测试KG编码器 ===
    print("\nTesting KG Encoder...")

    # 创建异构图
    x_dict = {
        'user': torch.randn(num_users, dim),
        'entity': torch.randn(num_entities, dim),
        'item': torch.randn(num_items, dim)
    }

    edge_index_dict = {
        ('user', 'long_term', 'entity'): torch.randint(0, min(num_users, num_entities), (2, 200)),
        ('user', 'short_term', 'entity'): torch.randint(0, min(num_users, num_entities), (2, 300)),
        ('entity', 'describes', 'item'): torch.randint(0, min(num_entities, num_items), (2, 150)),
    }

    kg_encoder = KGEncoder(dim, num_layers=2)
    out_dict = kg_encoder(x_dict, edge_index_dict)

    print(f"  User embedding: {out_dict['user'].shape}")
    print(f"  Entity embedding: {out_dict['entity'].shape}")
    print(f"  Item embedding: {out_dict['item'].shape}")

    print("\n✓ Encoders tested successfully!")
