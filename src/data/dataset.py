#!/usr/bin/env python3
"""
Dataset and DataLoader for recommendation

提供训练/验证/测试数据集，支持负采样。
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
    """推荐任务的Dataset"""

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
            interactions: 交互数据（包含user_id, item_id, rating等）
            user_id_map: User原始ID → 内部ID映射
            item_id_map: Item原始ID → 内部ID映射
            num_negatives: 负采样数量
            mode: 'train', 'val', 或 'test'
        """
        self.interactions = interactions
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.num_negatives = num_negatives
        self.mode = mode

        self.num_users = len(user_id_map)
        self.num_items = len(item_id_map)

        # 构建用户-物品交互字典（用于负采样）
        self.user_items = defaultdict(set)
        for _, row in interactions.iterrows():
            user_id = user_id_map[row['user_id:token']]
            item_id = item_id_map[row['item_id:token']]
            self.user_items[user_id].add(item_id)

        # 所有物品集合
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
                'neg_item_ids': List[int] (长度为num_negatives)
            }
        """
        row = self.interactions.iloc[idx]

        user_id = self.user_id_map[row['user_id:token']]
        pos_item_id = self.item_id_map[row['item_id:token']]

        # 负采样
        neg_item_ids = self._negative_sampling(user_id, self.num_negatives)

        return {
            'user_id': user_id,
            'pos_item_id': pos_item_id,
            'neg_item_ids': neg_item_ids
        }

    def _negative_sampling(self, user_id: int, num_neg: int) -> List[int]:
        """
        为用户采样负样本

        Args:
            user_id: 用户内部ID
            num_neg: 负样本数量

        Returns:
            neg_items: 负样本item IDs
        """
        pos_items = self.user_items[user_id]
        neg_candidates = list(self.all_items - pos_items)

        if len(neg_candidates) < num_neg:
            # 如果负候选不够，重复采样
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
    将batch整理成tensor

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
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分割数据集为训练/验证/测试集

    Args:
        inter_path: 交互文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        time_based: 是否基于时间分割（推荐）
        random_seed: 随机种子

    Returns:
        train_df, val_df, test_df
    """
    inter = pd.read_csv(inter_path, sep='\t')

    if time_based:
        # 基于时间戳排序
        inter = inter.sort_values('timestamp:float')

        # 按比例分割
        n = len(inter)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = inter.iloc[:train_end].reset_index(drop=True)
        val_df = inter.iloc[train_end:val_end].reset_index(drop=True)
        test_df = inter.iloc[val_end:].reset_index(drop=True)
    else:
        # 随机分割
        np.random.seed(random_seed)
        shuffled = inter.sample(frac=1, random_state=random_seed)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = shuffled.iloc[:train_end].reset_index(drop=True)
        val_df = shuffled.iloc[train_end:val_end].reset_index(drop=True)
        test_df = shuffled.iloc[val_end:].reset_index(drop=True)

    logger.info(f"Data split:")
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
    创建训练/验证/测试DataLoader

    Args:
        train_df, val_df, test_df: 数据集
        user_id_map, item_id_map: ID映射
        batch_size: batch大小
        num_negatives: 负采样数量
        num_workers: 数据加载进程数

    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建Dataset
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

    # 创建DataLoader
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
    # 测试用例
    logging.basicConfig(level=logging.INFO)

    # 首先需要构建图以获取ID映射
    from graph_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder(
        item_kg_path='data/recbole/ml-1m/ml-1m.item.kg',
        user_kg_path='data/recbole/ml-1m/ml-1m.user.kg',
        inter_path='data/recbole/ml-1m/ml-1m.inter'
    )

    _, _, stats = builder.build_hetero_graph()

    # 分割数据
    train_df, val_df, test_df = split_data(
        'data/recbole/ml-1m/ml-1m.inter',
        time_based=True
    )

    # 创建DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        stats['user_id_map'],
        stats['item_id_map'],
        batch_size=1024,
        num_negatives=1
    )

    # 测试一个batch
    batch = next(iter(train_loader))
    print(f"\n✓ Sample batch:")
    print(f"  user_id: {batch['user_id'].shape}")
    print(f"  pos_item_id: {batch['pos_item_id'].shape}")
    print(f"  neg_item_ids: {batch['neg_item_ids'].shape}")
