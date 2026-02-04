#!/usr/bin/env python3
"""
Graph Builder: Build heterogeneous graph (User-Entity-Item)

Build PyG HeteroData from RecBole-format KG files and interaction files.
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
    """Build heterogeneous graph for knowledge-enhanced recommendation"""

    def __init__(
        self,
        item_kg_path: str,
        user_kg_path: str,
        inter_path: str,
        min_rating: float = 4.0
    ):
        """
        Args:
            item_kg_path: Path to Item KG file (ml-1m.item.kg)
            user_kg_path: Path to User KG file (ml-1m.user.kg)
            inter_path: Path to interaction file (ml-1m.inter)
            min_rating: Minimum rating threshold (for filtering negative samples)
        """
        self.item_kg_path = Path(item_kg_path)
        self.user_kg_path = Path(user_kg_path)
        self.inter_path = Path(inter_path)
        self.min_rating = min_rating

        # Mapping tables (built during load_data)
        self.user_id_map = {}
        self.item_id_map = {}
        self.entity_id_map = {}

        # Reverse mappings
        self.id2user = {}
        self.id2item = {}
        self.id2entity = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data files"""
        logger.info("Loading data files...")

        # Read Item KG
        item_kg = pd.read_csv(
            self.item_kg_path,
            sep='\t',
            names=['head_id', 'relation_id', 'tail_id'],
            skiprows=1
        )
        # Drop rows with NaN values
        item_kg = item_kg.dropna()
        logger.info(f"  Item KG: {len(item_kg)} triplets")

        # Read User KG
        user_kg = pd.read_csv(
            self.user_kg_path,
            sep='\t',
            names=['head_id', 'relation_id', 'tail_id'],
            skiprows=1
        )
        # Drop rows with NaN values
        user_kg = user_kg.dropna()
        logger.info(f"  User KG: {len(user_kg)} triplets")

        # Read interaction data
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
        """Build ID mapping tables (original ID -> 0-based consecutive ID)"""
        logger.info("Building ID mappings...")

        # User ID mapping
        unique_users = sorted(inter['user_id:token'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.id2user = {idx: uid for uid, idx in self.user_id_map.items()}

        # Item ID mapping
        unique_items = sorted(inter['item_id:token'].unique())
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.id2item = {idx: iid for iid, idx in self.item_id_map.items()}

        # Entity ID mapping (from tail entities in Item KG and User KG)
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
        """Compute entity occurrence frequency (for mask initialization)"""
        entity_freq = Counter()

        # Entities in Item KG
        for entity in item_kg['tail_id']:
            entity_freq[entity] += 1

        # Entities in User KG
        for entity in user_kg['tail_id']:
            entity_freq[entity] += 1

        return entity_freq

    def build_hetero_graph(self, train_inter_df: Optional[pd.DataFrame] = None) -> Tuple[HeteroData, Dict]:
        """
        Build heterogeneous graph

        Args:
            train_inter_df: Training set interaction data (only used for building user-item edges and CF graph).
                           If None, uses the full inter file (causes data leakage!)

        Returns:
            hetero_graph: PyG HeteroData object
            stats: Statistics dictionary
        """
        # 1. Load data
        item_kg, user_kg, inter = self.load_data()

        # 2. Build ID mappings (need full inter to ensure coverage of all users and items)
        self.build_id_mappings(item_kg, user_kg, inter)

        # 3. Compute entity frequency
        entity_freq = self.compute_entity_frequency(item_kg, user_kg)

        # 4. Create HeteroData
        graph = HeteroData()

        # === Add nodes ===
        graph['user'].num_nodes = len(self.user_id_map)
        graph['entity'].num_nodes = len(self.entity_id_map)
        graph['item'].num_nodes = len(self.item_id_map)

        logger.info("Building heterogeneous graph...")

        # === 5. Add User-Entity edges (interests) ===
        self._add_user_entity_edges(graph, user_kg)

        # === 6. Add Entity-Item edges (descriptions) ===
        self._add_entity_item_edges(graph, item_kg)

        # === 7. Add User-Item edges (ratings) ===
        # Use training set data (if provided), otherwise use full data (causes leakage!)
        inter_for_edges = train_inter_df if train_inter_df is not None else inter
        if train_inter_df is not None:
            logger.info("  Using TRAIN-ONLY interactions for graph edges (no data leakage)")
        else:
            logger.warning("  WARNING: Using ALL interactions for graph edges (DATA LEAKAGE!)")
        self._add_user_item_edges(graph, inter_for_edges)

        # === 8. Build CF graph (User-Item bipartite graph) ===
        cf_graph = self._build_cf_graph(inter_for_edges)

        # === 9. Statistics ===
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

        # Add edge statistics
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
        """Add User-Entity edges (distinguishing long_term and short_term), including reverse edges"""
        # Separate two relation types
        long_term = user_kg[user_kg['relation_id'] == 'long_term_interest']
        short_term = user_kg[user_kg['relation_id'] == 'short_term_interest']

        # Long-term interest (bidirectional) - create edge type even if empty
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
            # Create empty edges (ensure model can handle them)
            graph['user', 'long_term', 'entity'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph['entity', 'rev_long_term', 'user'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            logger.info(f"  Added 0 long_term_interest edges (empty)")

        # Short-term interest (bidirectional) - create edge type even if empty
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
            # Create empty edges (ensure model can handle them)
            graph['user', 'short_term', 'entity'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph['entity', 'rev_short_term', 'user'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            logger.info(f"  Added 0 short_term_interest edges (empty)")

    def _add_entity_item_edges(self, graph: HeteroData, item_kg: pd.DataFrame):
        """Add Entity-Item edges (item visual features), including reverse edges"""
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
            # Create empty edges (ensure model can handle them)
            graph['entity', 'describes', 'item'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph['item', 'rev_describes', 'entity'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            logger.info(f"  Added 0 entity-item edges (empty)")

    def _add_user_item_edges(self, graph: HeteroData, inter: pd.DataFrame):
        """Add User-Item edges (rating interactions)"""
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
        Build CF graph (User-Item bipartite graph)

        Returns:
            edge_index: [2, num_edges] - edges with Users and Items under unified numbering
        """
        user_ids = [self.user_id_map[uid] for uid in inter['user_id:token']]
        # Item IDs need offset (in bipartite graph, Item nodes come after User nodes)
        item_ids = [
            self.item_id_map[iid] + len(self.user_id_map)
            for iid in inter['item_id:token']
        ]

        # Bidirectional edges (User->Item and Item->User)
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
    Compute initial mask based on entity frequency

    Frequency strategy:
    - Low frequency (<min_freq): mask=0.3 (possibly unreliable)
    - Moderate frequency: mask=1.0 (trustworthy)
    - High frequency (>max_freq): mask=0.8 (possibly too generic)

    Args:
        entity_freq: Counter object, original entity name -> frequency
        entity_id_map: Original entity name -> internal ID
        min_freq: Low frequency threshold
        max_freq: High frequency threshold

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
            # Low frequency: possibly noise or hallucination
            mask[entity_id] = 0.3
        elif freq > max_freq:
            # High frequency: possibly too generic (e.g. "colorful")
            mask[entity_id] = 0.8
        else:
            # Moderate frequency: trustworthy
            mask[entity_id] = 1.0

    logger.info(f"Mask initialization:")
    logger.info(f"  Low-freq entities (<{min_freq}): {(mask == 0.3).sum()}")
    logger.info(f"  High-freq entities (>{max_freq}): {(mask == 0.8).sum()}")
    logger.info(f"  Normal entities: {(mask == 1.0).sum()}")

    return mask


if __name__ == '__main__':
    # Test case
    logging.basicConfig(level=logging.INFO)

    builder = KnowledgeGraphBuilder(
        item_kg_path='data/ml-1m/ml-1m.item.kg',
        user_kg_path='data/ml-1m/ml-1m.user.kg',
        inter_path='data/ml-1m/ml-1m.inter'
    )

    hetero_graph, cf_graph, stats = builder.build_hetero_graph()

    # Compute initial mask
    mask_init = compute_frequency_mask(
        stats['entity_frequency'],
        stats['entity_id_map']
    )

    print(f"\n✓ Successfully built graphs!")
    print(f"  Hetero graph: {hetero_graph}")
    print(f"  CF graph edge_index shape: {cf_graph.shape}")
    print(f"  Initial mask shape: {mask_init.shape}")
