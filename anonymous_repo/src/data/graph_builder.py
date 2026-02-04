#!/usr/bin/env python3
"""
Graph Builder: 构建异构图（User-Entity-Item）

从RecBole格式的KG文件和交互文件构建PyG HeteroData。
"""

import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import Counter
from torch_geometric.data import HeteroData
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """构建知识增强推荐的异构图"""

    def __init__(
        self,
        item_kg_path: str,
        user_kg_path: str,
        inter_path: str,
        min_rating: float = 4.0
    ):
        """
        Args:
            item_kg_path: Item KG文件路径 (ml-1m.item.kg)
            user_kg_path: User KG文件路径 (ml-1m.user.kg)
            inter_path: 交互文件路径 (ml-1m.inter)
            min_rating: 最小评分阈值（用于过滤负样本）
        """
        self.item_kg_path = Path(item_kg_path)
        self.user_kg_path = Path(user_kg_path)
        self.inter_path = Path(inter_path)
        self.min_rating = min_rating

        # 映射表（在load_data时构建）
        self.user_id_map = {}
        self.item_id_map = {}
        self.entity_id_map = {}

        # 反向映射
        self.id2user = {}
        self.id2item = {}
        self.id2entity = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载所有数据文件"""
        logger.info("Loading data files...")

        # 读取Item KG
        item_kg = pd.read_csv(
            self.item_kg_path,
            sep='\t',
            names=['head_id', 'relation_id', 'tail_id'],
            skiprows=1
        )
        # Drop rows with NaN values
        item_kg = item_kg.dropna()
        logger.info(f"  Item KG: {len(item_kg)} triplets")

        # 读取User KG
        user_kg = pd.read_csv(
            self.user_kg_path,
            sep='\t',
            names=['head_id', 'relation_id', 'tail_id'],
            skiprows=1
        )
        # Drop rows with NaN values
        user_kg = user_kg.dropna()
        logger.info(f"  User KG: {len(user_kg)} triplets")

        # 读取交互数据
        inter = pd.read_csv(
            self.inter_path,
            sep='\t'
        )
        logger.info(f"  Interactions: {len(inter)} records")

        return item_kg, user_kg, inter

    def build_id_mappings(
        self,
        item_kg: pd.DataFrame,
        user_kg: pd.DataFrame,
        inter: pd.DataFrame
    ):
        """构建ID映射表（原始ID → 0-based连续ID）"""
        logger.info("Building ID mappings...")

        # User ID映射
        unique_users = sorted(inter['user_id:token'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.id2user = {idx: uid for uid, idx in self.user_id_map.items()}

        # Item ID映射
        unique_items = sorted(inter['item_id:token'].unique())
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.id2item = {idx: iid for iid, idx in self.item_id_map.items()}

        # Entity ID映射（来自Item KG和User KG的tail）
        # Filter out NaN and non-string values
        entities_from_item = set(
            e for e in item_kg['tail_id'].unique()
            if isinstance(e, str) and e.strip()
        )
        entities_from_user = set(
            e for e in user_kg['tail_id'].unique()
            if isinstance(e, str) and e.strip()
        )
        all_entities = sorted(entities_from_item | entities_from_user)

        self.entity_id_map = {eid: idx for idx, eid in enumerate(all_entities)}
        self.id2entity = {idx: eid for eid, idx in self.entity_id_map.items()}

        logger.info(f"  Users: {len(self.user_id_map)}")
        logger.info(f"  Items: {len(self.item_id_map)}")
        logger.info(f"  Entities: {len(self.entity_id_map)}")

    def compute_entity_frequency(
        self,
        item_kg: pd.DataFrame,
        user_kg: pd.DataFrame
    ) -> Counter:
        """统计Entity出现频率（用于Mask初始化）"""
        entity_freq = Counter()

        # Item KG中的entity
        for entity in item_kg['tail_id']:
            entity_freq[entity] += 1

        # User KG中的entity
        for entity in user_kg['tail_id']:
            entity_freq[entity] += 1

        return entity_freq

    def build_hetero_graph(self, train_inter_df: Optional[pd.DataFrame] = None) -> Tuple[HeteroData, Dict]:
        """
        构建异构图

        Args:
            train_inter_df: 训练集交互数据（仅用于构建user-item边和CF图）
                           如果为None，则使用完整的inter文件（会导致数据泄露！）

        Returns:
            hetero_graph: PyG HeteroData对象
            stats: 统计信息字典
        """
        # 1. 加载数据
        item_kg, user_kg, inter = self.load_data()

        # 2. 构建ID映射（需要用完整的inter来确保覆盖所有用户和物品）
        self.build_id_mappings(item_kg, user_kg, inter)

        # 3. 统计Entity频率
        entity_freq = self.compute_entity_frequency(item_kg, user_kg)

        # 4. 创建HeteroData
        graph = HeteroData()

        # === 添加节点 ===
        graph['user'].num_nodes = len(self.user_id_map)
        graph['entity'].num_nodes = len(self.entity_id_map)
        graph['item'].num_nodes = len(self.item_id_map)

        logger.info("Building heterogeneous graph...")

        # === 5. 添加User-Entity边（兴趣）===
        self._add_user_entity_edges(graph, user_kg)

        # === 6. 添加Entity-Item边（描述）===
        self._add_entity_item_edges(graph, item_kg)

        # === 7. 添加User-Item边（评分）===
        # 使用训练集数据（如果提供），否则使用完整数据（会泄露！）
        inter_for_edges = train_inter_df if train_inter_df is not None else inter
        if train_inter_df is not None:
            logger.info("  Using TRAIN-ONLY interactions for graph edges (no data leakage)")
        else:
            logger.warning("  WARNING: Using ALL interactions for graph edges (DATA LEAKAGE!)")
        self._add_user_item_edges(graph, inter_for_edges)

        # === 8. 构建CF图（User-Item二部图）===
        cf_graph = self._build_cf_graph(inter_for_edges)

        # === 9. 统计信息 ===
        stats = {
            'num_users': len(self.user_id_map),
            'num_items': len(self.item_id_map),
            'num_entities': len(self.entity_id_map),
            'entity_frequency': entity_freq,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'entity_id_map': self.entity_id_map,
            'id2user': self.id2user,
            'id2item': self.id2item,
            'id2entity': self.id2entity,
        }

        # 添加边统计
        for edge_type, edge_store in graph.edge_items():
            stats[f'{edge_type}_edges'] = edge_store.edge_index.size(1)

        logger.info("\nGraph statistics:")
        logger.info(f"  Nodes: {stats['num_users']} users, "
                   f"{stats['num_items']} items, "
                   f"{stats['num_entities']} entities")
        logger.info(f"  Edges:")
        for edge_type in graph.edge_types:
            logger.info(f"    {edge_type}: {stats[f'{edge_type}_edges']}")

        return graph, cf_graph, stats

    def _add_user_entity_edges(self, graph: HeteroData, user_kg: pd.DataFrame):
        """添加User-Entity边（区分long_term和short_term），包含反向边"""
        # 分离两种关系
        long_term = user_kg[user_kg['relation_id'] == 'long_term_interest']
        short_term = user_kg[user_kg['relation_id'] == 'short_term_interest']

        # Long-term interest (双向) - 即使为空也创建边类型
        if len(long_term) > 0:
            user_ids = [self.user_id_map[uid] for uid in long_term['head_id']]
            entity_ids = [self.entity_id_map[eid] for eid in long_term['tail_id']]

            # Forward edge: user -> entity
            edge_index = torch.tensor([user_ids, entity_ids], dtype=torch.long)
            graph['user', 'long_term', 'entity'].edge_index = edge_index

            # Reverse edge: entity -> user
            edge_index_rev = torch.tensor([entity_ids, user_ids], dtype=torch.long)
            graph['entity', 'rev_long_term', 'user'].edge_index = edge_index_rev

            logger.info(f"  Added {len(user_ids)} long_term_interest edges (bidirectional)")
        else:
            # 创建空边（确保模型能处理）
            graph['user', 'long_term', 'entity'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph['entity', 'rev_long_term', 'user'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            logger.info(f"  Added 0 long_term_interest edges (empty)")

        # Short-term interest (双向) - 即使为空也创建边类型
        if len(short_term) > 0:
            user_ids = [self.user_id_map[uid] for uid in short_term['head_id']]
            entity_ids = [self.entity_id_map[eid] for eid in short_term['tail_id']]

            # Forward edge: user -> entity
            edge_index = torch.tensor([user_ids, entity_ids], dtype=torch.long)
            graph['user', 'short_term', 'entity'].edge_index = edge_index

            # Reverse edge: entity -> user
            edge_index_rev = torch.tensor([entity_ids, user_ids], dtype=torch.long)
            graph['entity', 'rev_short_term', 'user'].edge_index = edge_index_rev

            logger.info(f"  Added {len(user_ids)} short_term_interest edges (bidirectional)")
        else:
            # 创建空边（确保模型能处理）
            graph['user', 'short_term', 'entity'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph['entity', 'rev_short_term', 'user'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            logger.info(f"  Added 0 short_term_interest edges (empty)")

    def _add_entity_item_edges(self, graph: HeteroData, item_kg: pd.DataFrame):
        """添加Entity-Item边（Item的视觉特征），包含反向边"""
        if len(item_kg) > 0:
            entity_ids = [self.entity_id_map[eid] for eid in item_kg['tail_id']]
            item_ids = [self.item_id_map[iid] for iid in item_kg['head_id']]

            # Forward edge: entity -> item
            edge_index = torch.tensor([entity_ids, item_ids], dtype=torch.long)
            graph['entity', 'describes', 'item'].edge_index = edge_index

            # Reverse edge: item -> entity
            edge_index_rev = torch.tensor([item_ids, entity_ids], dtype=torch.long)
            graph['item', 'rev_describes', 'entity'].edge_index = edge_index_rev

            logger.info(f"  Added {len(entity_ids)} entity-item edges (bidirectional)")
        else:
            # 创建空边（确保模型能处理）
            graph['entity', 'describes', 'item'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph['item', 'rev_describes', 'entity'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            logger.info(f"  Added 0 entity-item edges (empty)")

    def _add_user_item_edges(self, graph: HeteroData, inter: pd.DataFrame):
        """添加User-Item边（评分交互）"""
        user_ids = [self.user_id_map[uid] for uid in inter['user_id:token']]
        item_ids = [self.item_id_map[iid] for iid in inter['item_id:token']]
        ratings = inter['rating:float'].values

        edge_index = torch.tensor([user_ids, item_ids], dtype=torch.long)
        edge_attr = torch.tensor(ratings, dtype=torch.float)

        graph['user', 'rated', 'item'].edge_index = edge_index
        graph['user', 'rated', 'item'].edge_attr = edge_attr

        logger.info(f"  Added {len(user_ids)} user-item rating edges")

    def _build_cf_graph(self, inter: pd.DataFrame) -> torch.Tensor:
        """
        构建CF图（User-Item二部图）

        Returns:
            edge_index: [2, num_edges] - User和Item在统一编号下的边
        """
        user_ids = [self.user_id_map[uid] for uid in inter['user_id:token']]
        # Item ID需要偏移（在二部图中，Item节点在User节点之后）
        item_ids = [
            self.item_id_map[iid] + len(self.user_id_map)
            for iid in inter['item_id:token']
        ]

        # 双向边（User→Item 和 Item→User）
        edge_index = torch.tensor(
            [user_ids + item_ids, item_ids + user_ids],
            dtype=torch.long
        )

        logger.info(f"  Built CF graph with {edge_index.size(1)} edges (bidirectional)")

        return edge_index


def compute_frequency_mask(
    entity_freq: Counter,
    entity_id_map: Dict,
    min_freq: int = 5,
    max_freq: int = 1000
) -> torch.Tensor:
    """
    基于Entity频率计算初始Mask

    频率策略：
    - 低频（<min_freq）: mask=0.3（可能不可靠）
    - 适中频率：mask=1.0（可信）
    - 高频（>max_freq）: mask=0.8（可能太泛化）

    Args:
        entity_freq: Counter对象，原始entity名 -> 频率
        entity_id_map: 原始entity名 -> 内部ID
        min_freq: 低频阈值
        max_freq: 高频阈值

    Returns:
        mask: [num_entities] tensor
    """
    num_entities = len(entity_id_map)
    mask = torch.ones(num_entities)

    for entity_name, freq in entity_freq.items():
        if entity_name not in entity_id_map:
            continue

        entity_id = entity_id_map[entity_name]

        if freq < min_freq:
            # 低频：可能是噪声或幻觉
            mask[entity_id] = 0.3
        elif freq > max_freq:
            # 高频：可能太泛化（如"colorful"）
            mask[entity_id] = 0.8
        else:
            # 适中频率：可信
            mask[entity_id] = 1.0

    logger.info(f"Mask initialization:")
    logger.info(f"  Low-freq entities (<{min_freq}): {(mask == 0.3).sum()}")
    logger.info(f"  High-freq entities (>{max_freq}): {(mask == 0.8).sum()}")
    logger.info(f"  Normal entities: {(mask == 1.0).sum()}")

    return mask


if __name__ == '__main__':
    # 测试用例
    logging.basicConfig(level=logging.INFO)

    builder = KnowledgeGraphBuilder(
        item_kg_path='data/ml-1m/ml-1m.item.kg',
        user_kg_path='data/ml-1m/ml-1m.user.kg',
        inter_path='data/ml-1m/ml-1m.inter'
    )

    hetero_graph, cf_graph, stats = builder.build_hetero_graph()

    # 计算初始mask
    mask_init = compute_frequency_mask(
        stats['entity_frequency'],
        stats['entity_id_map']
    )

    print(f"\n✓ Successfully built graphs!")
    print(f"  Hetero graph: {hetero_graph}")
    print(f"  CF graph edge_index shape: {cf_graph.shape}")
    print(f"  Initial mask shape: {mask_init.shape}")
