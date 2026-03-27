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
from tqdm import tqdm

logger = logging.getLogger(__name__)


def ndcg_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    return_per_sample: bool = False
) -> Union[float, torch.Tensor]:
    """
    计算NDCG@K

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签（1=相关，0=不相关）
        k: Top-K
        return_per_sample: 是否返回每个样本的指标（True）或平均值（False）

    Returns:
        ndcg: NDCG@K平均值 (float) 或每个样本的NDCG (tensor [batch_size])
    """
    batch_size = scores.size(0)
    device = scores.device

    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # 批量计算DCG
    # 使用gather获取top-k标签
    top_k_labels = torch.gather(labels, 1, top_k_indices)  # [batch, k]

    # 计算DCG: sum((2^rel - 1) / log2(rank + 2))
    gains = 2.0 ** top_k_labels.float() - 1
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=device))
    dcg = (gains / discounts).sum(dim=1)  # [batch]

    # 批量计算IDCG
    ideal_labels, _ = torch.sort(labels, dim=1, descending=True)
    ideal_labels = ideal_labels[:, :k]  # [batch, k]
    ideal_gains = 2.0 ** ideal_labels.float() - 1
    idcg = (ideal_gains / discounts).sum(dim=1)  # [batch]

    # 计算NDCG（避免除零）
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))  # [batch]

    if return_per_sample:
        return ndcg
    else:
        return ndcg.mean().item()


def _dcg_at_k(labels: torch.Tensor) -> float:
    """计算DCG@K"""
    k = labels.size(0)
    gains = 2 ** labels.float() - 1
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=labels.device))
    return (gains / discounts).sum().item()


def recall_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    return_per_sample: bool = False
) -> Union[float, torch.Tensor]:
    """
    计算Recall@K

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签（1=相关，0=不相关）
        k: Top-K
        return_per_sample: 是否返回每个样本的指标（True）或平均值（False）

    Returns:
        recall: Recall@K平均值 (float) 或每个样本的Recall (tensor [batch_size])
    """
    batch_size = scores.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # 批量计算：使用gather获取top-k标签
    top_k_labels = torch.gather(labels, 1, top_k_indices)  # [batch, k]
    num_hit = top_k_labels.sum(dim=1)  # [batch]

    # 每个样本的相关物品总数
    num_relevant = labels.sum(dim=1)  # [batch]

    # Recall = Hit / Total Relevant（避免除零）
    recall = torch.where(num_relevant > 0, num_hit / num_relevant, torch.zeros_like(num_hit))  # [batch]

    if return_per_sample:
        return recall
    else:
        return recall.mean().item()


def precision_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    return_per_sample: bool = False
) -> Union[float, torch.Tensor]:
    """
    计算Precision@K

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签
        k: Top-K
        return_per_sample: 是否返回每个样本的指标（True）或平均值（False）

    Returns:
        precision: Precision@K平均值 (float) 或每个样本的Precision (tensor [batch_size])
    """
    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # 批量计算：使用gather获取top-k标签
    top_k_labels = torch.gather(labels, 1, top_k_indices)  # [batch, k]
    num_hit = top_k_labels.sum(dim=1)  # [batch]

    # Precision = Hit / K
    precision = num_hit / k  # [batch]

    if return_per_sample:
        return precision
    else:
        return precision.mean().item()


def hit_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    return_per_sample: bool = False
) -> Union[float, torch.Tensor]:
    """
    计算Hit@K（至少命中一个）

    Args:
        scores: [batch_size, num_items] - 预测分数
        labels: [batch_size, num_items] - 真实标签
        k: Top-K
        return_per_sample: 是否返回每个样本的指标（True）或平均值（False）

    Returns:
        hit_ratio: Hit@K比例 (float) 或每个样本的Hit (tensor [batch_size])
    """
    # 获取Top-K预测
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # 批量计算：使用gather获取top-k标签
    top_k_labels = torch.gather(labels, 1, top_k_indices)  # [batch, k]

    # Hit@K: 至少命中一个 (sum > 0)
    hit = (top_k_labels.sum(dim=1) > 0).float()  # [batch]

    if return_per_sample:
        return hit
    else:
        return hit.mean().item()


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


def evaluate_ranking_batched(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    test_user_items: Dict[int, List[int]],
    k_list: List[int] = [5, 10, 20],
    exclude_train: bool = True,
    train_user_items: Dict[int, List[int]] = None,
    val_user_items: Dict[int, List[int]] = None,
    mode: str = 'full',
    num_neg: int = 99,
    seed: int = 0,
    batch_size: int = 256
) -> Dict[str, float]:
    """
    批量并行Ranking评估（GPU加速版，充分利用A800算力）

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
        batch_size: 批量处理的用户数（default=256，A800可以设更大）

    Returns:
        metrics: 平均指标
    """
    device = user_emb.device
    num_items = item_emb.size(0)

    all_metrics = {f'{metric}@{k}': []
                   for metric in ['NDCG', 'Recall', 'Precision', 'Hit']
                   for k in k_list}

    # 设置随机种子
    if mode == 'uni100':
        random.seed(seed)
        np.random.seed(seed)

    if mode == 'uni100':
        # === uni100模式：批量并行评估 ===
        # 准备所有样本 (user_id, pos_item)
        eval_samples = []
        for user_id, test_items in test_user_items.items():
            if user_id >= user_emb.size(0):
                continue
            for pos_item in test_items:
                eval_samples.append((user_id, pos_item))

        if len(eval_samples) == 0:
            return {key: 0.0 for key in all_metrics.keys()}

        # 预先为每个用户准备负采样池
        user_candidate_items = {}
        for user_id in test_user_items.keys():
            if user_id >= user_emb.size(0):
                continue

            # 排除项目
            excluded_items = set(test_user_items.get(user_id, []))
            if exclude_train and train_user_items is not None:
                excluded_items.update(train_user_items.get(user_id, []))
            if val_user_items is not None:
                excluded_items.update(val_user_items.get(user_id, []))

            # 候选池
            all_items = np.arange(num_items)
            excluded_array = np.array(list(excluded_items))
            candidate_items = np.setdiff1d(all_items, excluded_array)

            if len(candidate_items) >= num_neg:
                user_candidate_items[user_id] = candidate_items

        # 过滤掉候选池不够的样本
        eval_samples = [(u, p) for u, p in eval_samples if u in user_candidate_items]

        if len(eval_samples) == 0:
            return {key: 0.0 for key in all_metrics.keys()}

        # 批量评估
        num_samples = len(eval_samples)
        pbar = tqdm(range(0, num_samples, batch_size), desc="Evaluating (batched)")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, num_samples)
            batch_samples = eval_samples[batch_start:batch_end]
            current_batch_size = len(batch_samples)

            # 准备batch数据
            batch_user_ids = []
            batch_pos_items = []
            batch_neg_items = []

            for user_id, pos_item in batch_samples:
                # 采样负样本
                neg_items = np.random.choice(
                    user_candidate_items[user_id],
                    size=num_neg,
                    replace=False
                )

                batch_user_ids.append(user_id)
                batch_pos_items.append(pos_item)
                batch_neg_items.append(neg_items)

            # 转换为tensor（优化：先转numpy再转tensor）
            batch_user_ids = torch.tensor(batch_user_ids, dtype=torch.long, device=device)
            batch_pos_items = torch.tensor(batch_pos_items, dtype=torch.long, device=device)
            # 优化：先转numpy array再转tensor（避免list of arrays警告）
            batch_neg_items = torch.from_numpy(np.array(batch_neg_items)).to(device)  # [batch, num_neg]

            # 批量获取embeddings
            batch_user_emb = user_emb[batch_user_ids]  # [batch, dim]
            batch_pos_emb = item_emb[batch_pos_items]  # [batch, dim]
            batch_neg_emb = item_emb[batch_neg_items]  # [batch, num_neg, dim]

            # 拼接 pos + neg
            batch_item_emb = torch.cat([
                batch_pos_emb.unsqueeze(1),  # [batch, 1, dim]
                batch_neg_emb  # [batch, num_neg, dim]
            ], dim=1)  # [batch, 1+num_neg, dim]

            # 批量计算分数
            # [batch, dim] @ [batch, 1+num_neg, dim].T -> [batch, 1+num_neg]
            batch_scores = torch.bmm(
                batch_user_emb.unsqueeze(1),  # [batch, 1, dim]
                batch_item_emb.transpose(1, 2)  # [batch, dim, 1+num_neg]
            ).squeeze(1)  # [batch, 1+num_neg]

            # 标签（第一个是正样本）
            batch_labels = torch.zeros_like(batch_scores)
            batch_labels[:, 0] = 1

            # 批量计算指标（向量化，无循环）
            for k in k_list:
                # 一次性计算整个batch的指标，返回 [batch_size] tensor
                batch_ndcg = ndcg_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_recall = recall_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_precision = precision_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_hit = hit_at_k(batch_scores, batch_labels, k, return_per_sample=True)

                # 转为CPU numpy并添加到列表（只在最后才传输到CPU）
                all_metrics[f'NDCG@{k}'].extend(batch_ndcg.cpu().numpy().tolist())
                all_metrics[f'Recall@{k}'].extend(batch_recall.cpu().numpy().tolist())
                all_metrics[f'Precision@{k}'].extend(batch_precision.cpu().numpy().tolist())
                all_metrics[f'Hit@{k}'].extend(batch_hit.cpu().numpy().tolist())

    else:
        # === full ranking模式：批量并行评估 ===
        user_ids = [uid for uid in test_user_items.keys() if uid < user_emb.size(0)]

        if len(user_ids) == 0:
            return {key: 0.0 for key in all_metrics.keys()}

        pbar = tqdm(range(0, len(user_ids), batch_size), desc="Evaluating (batched)")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(user_ids))
            batch_user_ids = user_ids[batch_start:batch_end]
            current_batch_size = len(batch_user_ids)

            # 批量计算分数 [batch, dim] @ [dim, num_items] -> [batch, num_items]
            batch_user_emb = user_emb[batch_user_ids]
            batch_scores = batch_user_emb @ item_emb.T  # [batch, num_items]

            # 构建标签和排除mask
            batch_labels = torch.zeros_like(batch_scores)

            for i, user_id in enumerate(batch_user_ids):
                # 设置正样本标签
                test_items = test_user_items[user_id]
                for item_id in test_items:
                    if item_id < num_items:
                        batch_labels[i, item_id] = 1

                # 排除训练集物品
                if exclude_train and train_user_items is not None:
                    train_items = train_user_items.get(user_id, [])
                    for item_id in train_items:
                        if item_id < num_items:
                            batch_scores[i, item_id] = float('-inf')

                # 排除验证集物品
                if val_user_items is not None:
                    val_items = val_user_items.get(user_id, [])
                    for item_id in val_items:
                        if item_id < num_items:
                            batch_scores[i, item_id] = float('-inf')

            # 批量计算指标（完全向量化，无循环）
            for k in k_list:
                # 一次性计算整个batch的指标，返回 [batch_size] tensor
                batch_ndcg = ndcg_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_recall = recall_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_precision = precision_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_hit = hit_at_k(batch_scores, batch_labels, k, return_per_sample=True)

                # 转为CPU numpy并添加到列表（只在最后才传输到CPU）
                all_metrics[f'NDCG@{k}'].extend(batch_ndcg.cpu().numpy().tolist())
                all_metrics[f'Recall@{k}'].extend(batch_recall.cpu().numpy().tolist())
                all_metrics[f'Precision@{k}'].extend(batch_precision.cpu().numpy().tolist())
                all_metrics[f'Hit@{k}'].extend(batch_hit.cpu().numpy().tolist())

    # 平均所有指标
    avg_metrics = {key: np.mean(values) if len(values) > 0 else 0.0
                   for key, values in all_metrics.items()}

    return avg_metrics


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

    ⚠️  WARNING: 这是旧版串行实现，推荐使用 evaluate_ranking_batched() 获得更好性能！

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
    # 自动切换到批量版本
    logger.warning("⚠️  使用旧版 evaluate_ranking，自动切换到批量优化版本 evaluate_ranking_batched")
    return evaluate_ranking_batched(
        user_emb, item_emb, test_user_items, k_list,
        exclude_train, train_user_items, val_user_items,
        mode, num_neg, seed, batch_size=256
    )
    all_metrics = {f'{metric}@{k}': []
                   for metric in ['NDCG', 'Recall', 'Precision', 'Hit']
                   for k in k_list}

    num_items = item_emb.size(0)

    # uni100模式下设置随机种子以保证可复现性
    if mode == 'uni100':
        random.seed(seed)
        np.random.seed(seed)

    # 添加进度条（uni100模式下评估较慢）
    user_items_iter = test_user_items.items()
    if mode == 'uni100':
        user_items_iter = tqdm(user_items_iter, desc="Evaluating", total=len(test_user_items))

    for user_id, test_items in user_items_iter:
        if user_id >= user_emb.size(0):
            continue

        if mode == 'uni100':
            # uni100 模式：对每个正样本分别评估（1 pos + num_neg negatives）
            if len(test_items) == 0:
                continue

            # 负采样池：排除训练集、验证集和测试集物品
            excluded_items = set(test_items)
            if exclude_train and train_user_items is not None:
                excluded_items.update(train_user_items.get(user_id, []))
            # CRITICAL: 评估测试集时，也要排除验证集物品
            if val_user_items is not None:
                excluded_items.update(val_user_items.get(user_id, []))

            # 候选负样本池 (优化：用 numpy 数组更快)
            all_items = np.arange(num_items)
            excluded_array = np.array(list(excluded_items))
            candidate_items = np.setdiff1d(all_items, excluded_array)
            if len(candidate_items) < num_neg:
                continue  # 候选池不够，跳过这个用户

            # 对每个测试集物品分别评估
            for pos_item in test_items:
                # 随机采样 num_neg 个负样本 (使用numpy更快)
                neg_items = np.random.choice(candidate_items, size=num_neg, replace=False).tolist()

                # 构建候选集：1 pos + num_neg neg
                eval_items = [pos_item] + neg_items

                # 计算分数
                eval_item_emb = item_emb[eval_items]  # [num_neg+1, dim]
                scores = (user_emb[user_id] @ eval_item_emb.T).cpu()  # [num_neg+1]

                # 构建标签（第一个是正样本）
                labels = torch.zeros(len(eval_items))
                labels[0] = 1

                # 计算指标（注意：要在循环内计算每个正样本的指标）
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

            # uni100模式下已经在上面计算过指标了，跳过后面的计算
            continue

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
