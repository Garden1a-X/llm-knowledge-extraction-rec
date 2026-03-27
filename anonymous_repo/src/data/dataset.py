#!/usr/bin/env python3
"""
Dataset and DataLoader for recommendation

Provides train/validation/test datasets with negative sampling support.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RecDataset(Dataset):
    """Dataset for recommendation tasks"""

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_id_map: Dict,
        item_id_map: Dict,
        num_negatives: int = 1,
        mode: str = 'train'
    ):
        """
        Args:
            interactions: Interaction data (contains user_id, item_id, rating, etc.)
            user_id_map: Mapping from original user ID to internal ID
            item_id_map: Mapping from original item ID to internal ID
            num_negatives: Number of negative samples
            mode: 'train', 'val', or 'test'
        """
        self.interactions = interactions
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.num_negatives = num_negatives
        self.mode = mode

        self.num_users = len(user_id_map)
        self.num_items = len(item_id_map)

        # Build user-item interaction dict (for negative sampling)
        self.user_items = defaultdict(set)
        for _, row in interactions.iterrows():
            user_id = user_id_map[row['user_id:token']]
            item_id = item_id_map[row['item_id:token']]
            self.user_items[user_id].add(item_id)

        # Set of all items
        self.all_items = set(range(self.num_items))

        logger.info(f"Created {mode} dataset: {len(interactions)} interactions")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        """
        Returns:
            {
                'user_id': int,
                'pos_item_id': int,
                'neg_item_ids': List[int] (length equals num_negatives)
            }
        """
        row = self.interactions.iloc[idx]

        user_id = self.user_id_map[row['user_id:token']]
        pos_item_id = self.item_id_map[row['item_id:token']]

        # Negative sampling
        neg_item_ids = self._negative_sampling(user_id, self.num_negatives)

        return {
            'user_id': user_id,
            'pos_item_id': pos_item_id,
            'neg_item_ids': neg_item_ids
        }

    def _negative_sampling(self, user_id: int, num_neg: int) -> List[int]:
        """
        Sample negative items for a user

        Args:
            user_id: Internal user ID
            num_neg: Number of negative samples

        Returns:
            neg_items: Negative sample item IDs
        """
        pos_items = self.user_items[user_id]
        neg_candidates = list(self.all_items - pos_items)

        if len(neg_candidates) < num_neg:
            # If not enough negative candidates, sample with replacement
            neg_items = np.random.choice(
                neg_candidates,
                size=num_neg,
                replace=True
            ).tolist()
        else:
            neg_items = np.random.choice(
                neg_candidates,
                size=num_neg,
                replace=False
            ).tolist()

        return neg_items


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a batch into tensors

    Args:
        batch: List of dicts from Dataset.__getitem__

    Returns:
        {
            'user_id': [batch_size],
            'pos_item_id': [batch_size],
            'neg_item_ids': [batch_size, num_negatives]
        }
    """
    user_ids = [item['user_id'] for item in batch]
    pos_item_ids = [item['pos_item_id'] for item in batch]
    neg_item_ids = [item['neg_item_ids'] for item in batch]

    return {
        'user_id': torch.tensor(user_ids, dtype=torch.long),
        'pos_item_id': torch.tensor(pos_item_ids, dtype=torch.long),
        'neg_item_ids': torch.tensor(neg_item_ids, dtype=torch.long)
    }


def split_data(
    inter_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    time_based: bool = True,
    random_seed: int = 42,
    per_user_split: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test sets (aligned with RecBole's RS split)

    Args:
        inter_path: Path to interaction file
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        time_based: Whether to sort by time first (per user)
        random_seed: Random seed
        per_user_split: Whether to use per-user split (recommended, aligned with RecBole RS)

    Returns:
        train_df, val_df, test_df
    """
    inter = pd.read_csv(inter_path, sep='\t')

    np.random.seed(random_seed)

    if per_user_split:
        # Per-user Random Split (aligned with RecBole's RS strategy)
        # Split each user's interaction sequence individually as 70/10/20

        train_list = []
        val_list = []
        test_list = []

        for user_id, user_inter in inter.groupby('user_id:token'):
            # Sort this user's interactions by time
            if time_based:
                user_inter = user_inter.sort_values('timestamp:float')
            else:
                user_inter = user_inter.sample(frac=1, random_state=random_seed)

            n = len(user_inter)

            # Need at least 3 interactions to split (1 for each of train/val/test)
            if n < 3:
                # Fewer than 3: put all into training set
                train_list.append(user_inter)
                continue

            # Split by ratio
            train_end = max(1, int(n * train_ratio))
            val_end = min(n - 1, train_end + max(1, int(n * val_ratio)))

            train_list.append(user_inter.iloc[:train_end])
            val_list.append(user_inter.iloc[train_end:val_end])
            test_list.append(user_inter.iloc[val_end:])

        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame(columns=inter.columns)
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=inter.columns)

    else:
        # Global split (legacy, not recommended)
        if time_based:
            inter = inter.sort_values('timestamp:float')
        else:
            inter = inter.sample(frac=1, random_state=random_seed)

        n = len(inter)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = inter.iloc[:train_end].reset_index(drop=True)
        val_df = inter.iloc[train_end:val_end].reset_index(drop=True)
        test_df = inter.iloc[val_end:].reset_index(drop=True)

    logger.info(f"Data split (per_user={per_user_split}):")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(inter)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/len(inter)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/len(inter)*100:.1f}%)")

    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_id_map: Dict,
    item_id_map: Dict,
    batch_size: int = 1024,
    num_negatives: int = 1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test DataLoaders

    Args:
        train_df, val_df, test_df: Datasets
        user_id_map, item_id_map: ID mappings
        batch_size: Batch size
        num_negatives: Number of negative samples
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create Datasets
    train_dataset = RecDataset(
        train_df, user_id_map, item_id_map,
        num_negatives=num_negatives,
        mode='train'
    )

    val_dataset = RecDataset(
        val_df, user_id_map, item_id_map,
        num_negatives=num_negatives,
        mode='val'
    )

    test_dataset = RecDataset(
        test_df, user_id_map, item_id_map,
        num_negatives=num_negatives,
        mode='test'
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    logger.info(f"Created DataLoaders:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test case
    logging.basicConfig(level=logging.INFO)

    # First build graph to get ID mappings
    from graph_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder(
        item_kg_path='data/ml-1m/ml-1m.item.kg',
        user_kg_path='data/ml-1m/ml-1m.user.kg',
        inter_path='data/ml-1m/ml-1m.inter'
    )

    _, _, stats = builder.build_hetero_graph()

    # Split data
    train_df, val_df, test_df = split_data(
        'data/ml-1m/ml-1m.inter',
        time_based=True
    )

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        stats['user_id_map'],
        stats['item_id_map'],
        batch_size=1024,
        num_negatives=1
    )

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\n✓ Sample batch:")
    print(f"  user_id: {batch['user_id'].shape}")
    print(f"  pos_item_id: {batch['pos_item_id'].shape}")
    print(f"  neg_item_ids: {batch['neg_item_ids'].shape}")
