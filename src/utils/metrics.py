#!/usr/bin/env python3
"""
Evaluation metrics for recommendation

实现NDCG、Recall、Precision等推荐指标。
"""

import torch
import numpy as np
import random
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)


def ndcg_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10
) -> float:
    """
    计算NDCG@K

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签（1=相关，0=不相关）
        k: Top-K

    Returns:
        ndcg: NDCG@K平均值
    """
    batch_size = scores.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    ndcg_sum = 0.0

    for i in range(batch_size):
        # 获取Top-K的标签
        top_k_labels = labels[i, top_k_indices[i]]

        # DCG@K
        dcg = _dcg_at_k(top_k_labels)

        # IDCG@K（理想情况：按相关性排序）
        ideal_labels, _ = torch.sort(labels[i], descending=True)
        idcg = _dcg_at_k(ideal_labels[:k])

        # NDCG
        if idcg > 0:
            ndcg_sum += dcg / idcg

    return ndcg_sum / batch_size


def _dcg_at_k(labels: torch.Tensor) -> float:
    """计算DCG@K"""
    k = labels.size(0)
    gains = 2 ** labels.float() - 1
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float))
    return (gains / discounts).sum().item()


def recall_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10
) -> float:
    """
    计算Recall@K

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签（1=相关，0=不相关）
        k: Top-K

    Returns:
        recall: Recall@K平均值
    """
    batch_size = scores.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    recall_sum = 0.0

    for i in range(batch_size):
        # 真实相关物品数
        num_relevant = labels[i].sum().item()

        if num_relevant == 0:
            continue

        # Top-K中的相关物品数
        top_k_labels = labels[i, top_k_indices[i]]
        num_hit = top_k_labels.sum().item()

        # Recall = Hit / Total Relevant
        recall_sum += num_hit / num_relevant

    return recall_sum / batch_size


def precision_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10
) -> float:
    """
    计算Precision@K

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签
        k: Top-K

    Returns:
        precision: Precision@K平均值
    """
    batch_size = scores.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    precision_sum = 0.0

    for i in range(batch_size):
        # Top-K中的相关物品数
        top_k_labels = labels[i, top_k_indices[i]]
        num_hit = top_k_labels.sum().item()

        # Precision = Hit / K
        precision_sum += num_hit / k

    return precision_sum / batch_size


def hit_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10
) -> float:
    """
    计算Hit@K（至少命中一个）

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签
        k: Top-K

    Returns:
        hit_ratio: Hit@K比例
    """
    batch_size = scores.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    hit_count = 0

    for i in range(batch_size):
        # Top-K中是否有相关物品
        top_k_labels = labels[i, top_k_indices[i]]
        if top_k_labels.sum().item() > 0:
            hit_count += 1

    return hit_count / batch_size


def evaluate_all_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签
        k_list: K值列表

    Returns:
        metrics: 所有指标字典
    """
    metrics = {}

    for k in k_list:
        metrics[f'NDCG@{k}'] = ndcg_at_k(scores, labels, k)
        metrics[f'Recall@{k}'] = recall_at_k(scores, labels, k)
        metrics[f'Precision@{k}'] = precision_at_k(scores, labels, k)
        metrics[f'Hit@{k}'] = hit_at_k(scores, labels, k)

    return metrics


def evaluate_ranking(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    test_user_items: Dict[int, List[int]],
    k_list: List[int] = [5, 10, 20],
    exclude_train: bool = True,
    train_user_items: Dict[int, List[int]] = None,
    val_user_items: Dict[int, List[int]] = None,
    mode: str = 'full',
    num_neg: int = 99,
    seed: int = 0
) -> Dict[str, float]:
    """
    Ranking评估（支持full ranking和负采样）

    Args:
        user_emb: [num_users, dim] - 用户embedding
        item_emb: [num_items, dim] - 物品embedding
        test_user_items: 测试集中每个用户的正样本物品
        k_list: K值列表
        exclude_train: 是否排除训练集物品
        train_user_items: 训练集中每个用户的物品（用于排除）
        val_user_items: 验证集中每个用户的物品（用于排除，评估测试集时必须提供）
        mode: 'full' 或 'uni100' - 评估模式
        num_neg: 负采样数量（mode='uni100'时使用，默认99）
        seed: 随机种子（用于uni100负采样的可复现性，默认0）

    Returns:
        metrics: 平均指标
    """
    all_metrics = {f'{metric}@{k}': []
                   for metric in ['NDCG', 'Recall', 'Precision', 'Hit']
                   for k in k_list}

    num_items = item_emb.size(0)

    # uni100模式下设置随机种子以保证可复现性
    if mode == 'uni100':
        random.seed(seed)
        np.random.seed(seed)

    for user_id, test_items in test_user_items.items():
        if user_id >= user_emb.size(0):
            continue

        if mode == 'uni100':
            # uni100 模式：1 positive + num_neg random negatives
            # 选择一个正样本
            if len(test_items) == 0:
                continue
            pos_item = test_items[0]  # 取第一个正样本

            # 负采样：排除训练集、验证集和测试集物品
            excluded_items = set(test_items)
            if exclude_train and train_user_items is not None:
                excluded_items.update(train_user_items.get(user_id, []))
            # CRITICAL: 评估测试集时，也要排除验证集物品
            if val_user_items is not None:
                excluded_items.update(val_user_items.get(user_id, []))

            # 候选负样本池
            candidate_items = [i for i in range(num_items) if i not in excluded_items]
            if len(candidate_items) < num_neg:
                continue  # 候选池不够，跳过这个用户

            # 随机采样 num_neg 个负样本
            neg_items = random.sample(candidate_items, num_neg)

            # 构建候选集：1 pos + num_neg neg
            eval_items = [pos_item] + neg_items

            # 计算分数
            eval_item_emb = item_emb[eval_items]  # [num_neg+1, dim]
            scores = (user_emb[user_id] @ eval_item_emb.T).cpu()  # [num_neg+1]

            # 构建标签（第一个是正样本）
            labels = torch.zeros(len(eval_items))
            labels[0] = 1

        else:
            # full ranking 模式（原逻辑）
            # 计算该用户对所有物品的分数
            scores = (user_emb[user_id] @ item_emb.T).cpu()  # [num_items]

            # 构建标签
            labels = torch.zeros(num_items)
            for item_id in test_items:
                if item_id < num_items:
                    labels[item_id] = 1

            # 排除训练集物品（将分数设为-inf）
            if exclude_train and train_user_items is not None:
                train_items = train_user_items.get(user_id, [])
                for item_id in train_items:
                    if item_id < num_items:
                        scores[item_id] = float('-inf')

        # 计算指标
        for k in k_list:
            all_metrics[f'NDCG@{k}'].append(
                ndcg_at_k(scores.unsqueeze(0), labels.unsqueeze(0), k)
            )
            all_metrics[f'Recall@{k}'].append(
                recall_at_k(scores.unsqueeze(0), labels.unsqueeze(0), k)
            )
            all_metrics[f'Precision@{k}'].append(
                precision_at_k(scores.unsqueeze(0), labels.unsqueeze(0), k)
            )
            all_metrics[f'Hit@{k}'].append(
                hit_at_k(scores.unsqueeze(0), labels.unsqueeze(0), k)
            )

    # 平均所有用户的指标
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    return avg_metrics


if __name__ == '__main__':
    # 测试用例
    torch.manual_seed(42)

    # 模拟数据
    batch_size = 4
    num_items = 100

    # 随机分数和标签
    scores = torch.randn(batch_size, num_items)
    labels = torch.zeros(batch_size, num_items)

    # 每个用户有2-5个相关物品
    for i in range(batch_size):
        num_relevant = np.random.randint(2, 6)
        relevant_items = np.random.choice(num_items, num_relevant, replace=False)
        labels[i, relevant_items] = 1

    # 计算指标
    metrics = evaluate_all_metrics(scores, labels, k_list=[5, 10, 20])

    print("✓ Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
