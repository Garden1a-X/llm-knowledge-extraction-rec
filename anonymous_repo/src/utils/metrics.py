#!/usr/bin/env python3
"""
Evaluation metrics for recommendation

Implements recommendation metrics including NDCG, Recall, Precision, etc.
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
    Compute NDCG@K

    Args:
        scores: [batch_size, num_items] - predicted scores
        labels: [batch_size, num_items] - ground truth labels (1=relevant, 0=irrelevant)
        k: Top-K
        return_per_sample: whether to return per-sample metrics (True) or the average (False)

    Returns:
        ndcg: NDCG@K average (float) or per-sample NDCG (tensor [batch_size])
    """
    batch_size = scores.size(0)
    device = scores.device

    # Get Top-K predictions
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # Batch DCG computation
    # Use gather to obtain top-k labels
    top_k_labels = torch.gather(labels, 1, top_k_indices)  # [batch, k]

    # Compute DCG: sum((2^rel - 1) / log2(rank + 2))
    gains = 2.0 ** top_k_labels.float() - 1
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=device))
    dcg = (gains / discounts).sum(dim=1)  # [batch]

    # Batch IDCG computation
    ideal_labels, _ = torch.sort(labels, dim=1, descending=True)
    ideal_labels = ideal_labels[:, :k]  # [batch, k]
    ideal_gains = 2.0 ** ideal_labels.float() - 1
    idcg = (ideal_gains / discounts).sum(dim=1)  # [batch]

    # Compute NDCG (avoid division by zero)
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))  # [batch]

    if return_per_sample:
        return ndcg
    else:
        return ndcg.mean().item()


def _dcg_at_k(labels: torch.Tensor) -> float:
    """Compute DCG@K"""
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
    Compute Recall@K

    Args:
        scores: [batch_size, num_items] - predicted scores
        labels: [batch_size, num_items] - ground truth labels (1=relevant, 0=irrelevant)
        k: Top-K
        return_per_sample: whether to return per-sample metrics (True) or the average (False)

    Returns:
        recall: Recall@K average (float) or per-sample Recall (tensor [batch_size])
    """
    batch_size = scores.size(0)

    # Get Top-K predictions
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # Batch computation: use gather to obtain top-k labels
    top_k_labels = torch.gather(labels, 1, top_k_indices)  # [batch, k]
    num_hit = top_k_labels.sum(dim=1)  # [batch]

    # Total number of relevant items per sample
    num_relevant = labels.sum(dim=1)  # [batch]

    # Recall = Hit / Total Relevant (avoid division by zero)
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
    Compute Precision@K

    Args:
        scores: [batch_size, num_items] - predicted scores
        labels: [batch_size, num_items] - ground truth labels
        k: Top-K
        return_per_sample: whether to return per-sample metrics (True) or the average (False)

    Returns:
        precision: Precision@K average (float) or per-sample Precision (tensor [batch_size])
    """
    # Get Top-K predictions
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # Batch computation: use gather to obtain top-k labels
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
    Compute Hit@K (at least one hit)

    Args:
        scores: [batch_size, num_items] - predicted scores
        labels: [batch_size, num_items] - ground truth labels
        k: Top-K
        return_per_sample: whether to return per-sample metrics (True) or the average (False)

    Returns:
        hit_ratio: Hit@K ratio (float) or per-sample Hit (tensor [batch_size])
    """
    # Get Top-K predictions
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # Batch computation: use gather to obtain top-k labels
    top_k_labels = torch.gather(labels, 1, top_k_indices)  # [batch, k]

    # Hit@K: at least one hit (sum > 0)
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
    Compute all evaluation metrics

    Args:
        scores: [batch_size, num_items] - predicted scores
        labels: [batch_size, num_items] - ground truth labels
        k_list: list of K values

    Returns:
        metrics: dictionary of all metrics
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
    Batched parallel ranking evaluation (GPU-accelerated, fully utilizing GPU compute)

    Args:
        user_emb: [num_users, dim] - user embeddings
        item_emb: [num_items, dim] - item embeddings
        test_user_items: positive items per user in the test set
        k_list: list of K values
        exclude_train: whether to exclude training set items
        train_user_items: items per user in the training set (for exclusion)
        val_user_items: items per user in the validation set (for exclusion; must be provided when evaluating on the test set)
        mode: 'full' or 'uni100' - evaluation mode
        num_neg: number of negative samples (used when mode='uni100', default 99)
        seed: random seed (for reproducibility in uni100 negative sampling, default 0)
        batch_size: number of users per batch (default=256, can be set larger for high-end GPUs)

    Returns:
        metrics: averaged metrics
    """
    device = user_emb.device
    num_items = item_emb.size(0)

    all_metrics = {f'{metric}@{k}': []
                   for metric in ['NDCG', 'Recall', 'Precision', 'Hit']
                   for k in k_list}

    # Set random seeds
    if mode == 'uni100':
        random.seed(seed)
        np.random.seed(seed)

    if mode == 'uni100':
        # === uni100 mode: batched parallel evaluation ===
        # Prepare all samples (user_id, pos_item)
        eval_samples = []
        for user_id, test_items in test_user_items.items():
            if user_id >= user_emb.size(0):
                continue
            for pos_item in test_items:
                eval_samples.append((user_id, pos_item))

        if len(eval_samples) == 0:
            return {key: 0.0 for key in all_metrics.keys()}

        # Pre-build negative sampling pool for each user
        user_candidate_items = {}
        for user_id in test_user_items.keys():
            if user_id >= user_emb.size(0):
                continue

            # Items to exclude
            excluded_items = set(test_user_items.get(user_id, []))
            if exclude_train and train_user_items is not None:
                excluded_items.update(train_user_items.get(user_id, []))
            if val_user_items is not None:
                excluded_items.update(val_user_items.get(user_id, []))

            # Candidate pool
            all_items = np.arange(num_items)
            excluded_array = np.array(list(excluded_items))
            candidate_items = np.setdiff1d(all_items, excluded_array)

            if len(candidate_items) >= num_neg:
                user_candidate_items[user_id] = candidate_items

        # Filter out samples whose candidate pool is too small
        eval_samples = [(u, p) for u, p in eval_samples if u in user_candidate_items]

        if len(eval_samples) == 0:
            return {key: 0.0 for key in all_metrics.keys()}

        # Batched evaluation
        num_samples = len(eval_samples)
        pbar = tqdm(range(0, num_samples, batch_size), desc="Evaluating (batched)")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, num_samples)
            batch_samples = eval_samples[batch_start:batch_end]
            current_batch_size = len(batch_samples)

            # Prepare batch data
            batch_user_ids = []
            batch_pos_items = []
            batch_neg_items = []

            for user_id, pos_item in batch_samples:
                # Sample negative items
                neg_items = np.random.choice(
                    user_candidate_items[user_id],
                    size=num_neg,
                    replace=False
                )

                batch_user_ids.append(user_id)
                batch_pos_items.append(pos_item)
                batch_neg_items.append(neg_items)

            # Convert to tensors (optimization: convert to numpy first, then to tensor)
            batch_user_ids = torch.tensor(batch_user_ids, dtype=torch.long, device=device)
            batch_pos_items = torch.tensor(batch_pos_items, dtype=torch.long, device=device)
            # Optimization: convert to numpy array first, then to tensor (avoids list-of-arrays warning)
            batch_neg_items = torch.from_numpy(np.array(batch_neg_items)).to(device)  # [batch, num_neg]

            # Batch retrieve embeddings
            batch_user_emb = user_emb[batch_user_ids]  # [batch, dim]
            batch_pos_emb = item_emb[batch_pos_items]  # [batch, dim]
            batch_neg_emb = item_emb[batch_neg_items]  # [batch, num_neg, dim]

            # Concatenate pos + neg
            batch_item_emb = torch.cat([
                batch_pos_emb.unsqueeze(1),  # [batch, 1, dim]
                batch_neg_emb  # [batch, num_neg, dim]
            ], dim=1)  # [batch, 1+num_neg, dim]

            # Batch score computation
            # [batch, dim] @ [batch, 1+num_neg, dim].T -> [batch, 1+num_neg]
            batch_scores = torch.bmm(
                batch_user_emb.unsqueeze(1),  # [batch, 1, dim]
                batch_item_emb.transpose(1, 2)  # [batch, dim, 1+num_neg]
            ).squeeze(1)  # [batch, 1+num_neg]

            # Labels (the first item is the positive sample)
            batch_labels = torch.zeros_like(batch_scores)
            batch_labels[:, 0] = 1

            # Batch metric computation (vectorized, no loops)
            for k in k_list:
                # Compute metrics for the entire batch at once, returns [batch_size] tensor
                batch_ndcg = ndcg_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_recall = recall_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_precision = precision_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_hit = hit_at_k(batch_scores, batch_labels, k, return_per_sample=True)

                # Transfer to CPU numpy and append to list (only transfer to CPU at the end)
                all_metrics[f'NDCG@{k}'].extend(batch_ndcg.cpu().numpy().tolist())
                all_metrics[f'Recall@{k}'].extend(batch_recall.cpu().numpy().tolist())
                all_metrics[f'Precision@{k}'].extend(batch_precision.cpu().numpy().tolist())
                all_metrics[f'Hit@{k}'].extend(batch_hit.cpu().numpy().tolist())

    else:
        # === Full ranking mode: batched parallel evaluation ===
        user_ids = [uid for uid in test_user_items.keys() if uid < user_emb.size(0)]

        if len(user_ids) == 0:
            return {key: 0.0 for key in all_metrics.keys()}

        pbar = tqdm(range(0, len(user_ids), batch_size), desc="Evaluating (batched)")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(user_ids))
            batch_user_ids = user_ids[batch_start:batch_end]
            current_batch_size = len(batch_user_ids)

            # Batch score computation: [batch, dim] @ [dim, num_items] -> [batch, num_items]
            batch_user_emb = user_emb[batch_user_ids]
            batch_scores = batch_user_emb @ item_emb.T  # [batch, num_items]

            # Build labels and exclusion mask
            batch_labels = torch.zeros_like(batch_scores)

            for i, user_id in enumerate(batch_user_ids):
                # Set positive sample labels
                test_items = test_user_items[user_id]
                for item_id in test_items:
                    if item_id < num_items:
                        batch_labels[i, item_id] = 1

                # Exclude training set items
                if exclude_train and train_user_items is not None:
                    train_items = train_user_items.get(user_id, [])
                    for item_id in train_items:
                        if item_id < num_items:
                            batch_scores[i, item_id] = float('-inf')

                # Exclude validation set items
                if val_user_items is not None:
                    val_items = val_user_items.get(user_id, [])
                    for item_id in val_items:
                        if item_id < num_items:
                            batch_scores[i, item_id] = float('-inf')

            # Batch metric computation (fully vectorized, no loops)
            for k in k_list:
                # Compute metrics for the entire batch at once, returns [batch_size] tensor
                batch_ndcg = ndcg_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_recall = recall_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_precision = precision_at_k(batch_scores, batch_labels, k, return_per_sample=True)
                batch_hit = hit_at_k(batch_scores, batch_labels, k, return_per_sample=True)

                # Transfer to CPU numpy and append to list (only transfer to CPU at the end)
                all_metrics[f'NDCG@{k}'].extend(batch_ndcg.cpu().numpy().tolist())
                all_metrics[f'Recall@{k}'].extend(batch_recall.cpu().numpy().tolist())
                all_metrics[f'Precision@{k}'].extend(batch_precision.cpu().numpy().tolist())
                all_metrics[f'Hit@{k}'].extend(batch_hit.cpu().numpy().tolist())

    # Average all metrics
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
    Ranking evaluation (supports full ranking and negative sampling)

    WARNING: This is the legacy serial implementation. It is recommended to use evaluate_ranking_batched() for better performance!

    Args:
        user_emb: [num_users, dim] - user embeddings
        item_emb: [num_items, dim] - item embeddings
        test_user_items: positive items per user in the test set
        k_list: list of K values
        exclude_train: whether to exclude training set items
        train_user_items: items per user in the training set (for exclusion)
        val_user_items: items per user in the validation set (for exclusion; must be provided when evaluating on the test set)
        mode: 'full' or 'uni100' - evaluation mode
        num_neg: number of negative samples (used when mode='uni100', default 99)
        seed: random seed (for reproducibility in uni100 negative sampling, default 0)

    Returns:
        metrics: averaged metrics
    """
    # Automatically switch to the batched version
    logger.warning("WARNING: Using legacy evaluate_ranking, automatically switching to batched optimized version evaluate_ranking_batched")
    return evaluate_ranking_batched(
        user_emb, item_emb, test_user_items, k_list,
        exclude_train, train_user_items, val_user_items,
        mode, num_neg, seed, batch_size=256
    )
    all_metrics = {f'{metric}@{k}': []
                   for metric in ['NDCG', 'Recall', 'Precision', 'Hit']
                   for k in k_list}

    num_items = item_emb.size(0)

    # Set random seed for reproducibility in uni100 mode
    if mode == 'uni100':
        random.seed(seed)
        np.random.seed(seed)

    # Add progress bar (evaluation is slow in uni100 mode)
    user_items_iter = test_user_items.items()
    if mode == 'uni100':
        user_items_iter = tqdm(user_items_iter, desc="Evaluating", total=len(test_user_items))

    for user_id, test_items in user_items_iter:
        if user_id >= user_emb.size(0):
            continue

        if mode == 'uni100':
            # uni100 mode: evaluate each positive sample separately (1 pos + num_neg negatives)
            if len(test_items) == 0:
                continue

            # Negative sampling pool: exclude training, validation, and test set items
            excluded_items = set(test_items)
            if exclude_train and train_user_items is not None:
                excluded_items.update(train_user_items.get(user_id, []))
            # CRITICAL: When evaluating on the test set, also exclude validation set items
            if val_user_items is not None:
                excluded_items.update(val_user_items.get(user_id, []))

            # Candidate negative sample pool (optimization: numpy arrays are faster)
            all_items = np.arange(num_items)
            excluded_array = np.array(list(excluded_items))
            candidate_items = np.setdiff1d(all_items, excluded_array)
            if len(candidate_items) < num_neg:
                continue  # Candidate pool too small, skip this user

            # Evaluate each test item separately
            for pos_item in test_items:
                # Randomly sample num_neg negative items (using numpy for speed)
                neg_items = np.random.choice(candidate_items, size=num_neg, replace=False).tolist()

                # Build candidate set: 1 pos + num_neg neg
                eval_items = [pos_item] + neg_items

                # Compute scores
                eval_item_emb = item_emb[eval_items]  # [num_neg+1, dim]
                scores = (user_emb[user_id] @ eval_item_emb.T).cpu()  # [num_neg+1]

                # Build labels (the first item is the positive sample)
                labels = torch.zeros(len(eval_items))
                labels[0] = 1

                # Compute metrics (note: compute metrics for each positive sample within the loop)
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

            # In uni100 mode, metrics are already computed above; skip the computation below
            continue

        else:
            # Full ranking mode (original logic)
            # Compute scores for all items for this user
            scores = (user_emb[user_id] @ item_emb.T).cpu()  # [num_items]

            # Build labels
            labels = torch.zeros(num_items)
            for item_id in test_items:
                if item_id < num_items:
                    labels[item_id] = 1

            # Exclude training set items (set scores to -inf)
            if exclude_train and train_user_items is not None:
                train_items = train_user_items.get(user_id, [])
                for item_id in train_items:
                    if item_id < num_items:
                        scores[item_id] = float('-inf')

        # Compute metrics
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

    # Average metrics across all users
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    return avg_metrics


if __name__ == '__main__':
    # Test cases
    torch.manual_seed(42)

    # Simulated data
    batch_size = 4
    num_items = 100

    # Random scores and labels
    scores = torch.randn(batch_size, num_items)
    labels = torch.zeros(batch_size, num_items)

    # Each user has 2-5 relevant items
    for i in range(batch_size):
        num_relevant = np.random.randint(2, 6)
        relevant_items = np.random.choice(num_items, num_relevant, replace=False)
        labels[i, relevant_items] = 1

    # Compute metrics
    metrics = evaluate_all_metrics(scores, labels, k_list=[5, 10, 20])

    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
