"""
Evaluator for recommendation models.

Implements standard ranking metrics:
- NDCG@K: Normalized Discounted Cumulative Gain
- Recall@K: Recall at K
- Precision@K: Precision at K
- Hit@K: Hit rate at K
- MRR: Mean Reciprocal Rank
"""

import numpy as np
import torch
from typing import Dict, List, Set, Optional
from collections import defaultdict
import pandas as pd


class RecommendationEvaluator:
    """Evaluator for top-K recommendation."""

    def __init__(
        self,
        k_list: List[int] = [10, 20],
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize evaluator.

        Args:
            k_list: List of K values for metrics
            metrics: List of metrics to compute. If None, compute all.
                    Options: ['ndcg', 'recall', 'precision', 'hit', 'mrr']
        """
        self.k_list = k_list
        self.metrics = metrics or ['ndcg', 'recall', 'precision', 'hit', 'mrr']

    def evaluate(
        self,
        predictions: Dict[int, np.ndarray],
        ground_truth: Dict[int, Set[int]],
        train_data: Optional[Dict[int, Set[int]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations.

        Args:
            predictions: Dict mapping user_idx to array of predicted scores for all items
            ground_truth: Dict mapping user_idx to set of ground truth item indices
            train_data: Optional dict mapping user_idx to set of training items (to exclude)

        Returns:
            Dictionary of metric values
        """
        results = {}

        # Compute metrics for each user
        user_metrics = defaultdict(list)

        for user_idx, pred_scores in predictions.items():
            # Get ground truth
            true_items = ground_truth.get(user_idx, set())
            if len(true_items) == 0:
                continue

            # Exclude training items if provided
            if train_data is not None:
                train_items = train_data.get(user_idx, set())
                # Set training items to -inf so they don't appear in top-K
                pred_scores = pred_scores.copy()
                pred_scores[list(train_items)] = -np.inf

            # Get top-K recommendations
            # argsort returns indices in ascending order, so we reverse
            ranked_items = np.argsort(pred_scores)[::-1]

            # Compute metrics for each K
            for k in self.k_list:
                top_k = ranked_items[:k]

                if 'ndcg' in self.metrics:
                    ndcg = self._ndcg_at_k(top_k, true_items, k)
                    user_metrics[f'NDCG@{k}'].append(ndcg)

                if 'recall' in self.metrics:
                    recall = self._recall_at_k(top_k, true_items)
                    user_metrics[f'Recall@{k}'].append(recall)

                if 'precision' in self.metrics:
                    precision = self._precision_at_k(top_k, true_items)
                    user_metrics[f'Precision@{k}'].append(precision)

                if 'hit' in self.metrics:
                    hit = self._hit_at_k(top_k, true_items)
                    user_metrics[f'Hit@{k}'].append(hit)

            # MRR is not K-specific
            if 'mrr' in self.metrics:
                mrr = self._mrr(ranked_items, true_items)
                user_metrics['MRR'].append(mrr)

        # Average across users
        for metric_name, values in user_metrics.items():
            results[metric_name] = np.mean(values)

        return results

    def _ndcg_at_k(
        self,
        ranked_items: np.ndarray,
        true_items: Set[int],
        k: int
    ) -> float:
        """
        Compute NDCG@K.

        NDCG = DCG / IDCG
        DCG = sum(rel_i / log2(i + 2)) for i in 1..k
        """
        # Relevance: 1 if item in true_items, 0 otherwise
        relevance = np.array([1.0 if item in true_items else 0.0 for item in ranked_items])

        # DCG
        dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))

        # IDCG (ideal DCG)
        ideal_relevance = np.ones(min(len(true_items), k))
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _recall_at_k(
        self,
        ranked_items: np.ndarray,
        true_items: Set[int]
    ) -> float:
        """
        Compute Recall@K.

        Recall = |recommended ∩ relevant| / |relevant|
        """
        hits = len(set(ranked_items) & true_items)
        return hits / len(true_items) if len(true_items) > 0 else 0.0

    def _precision_at_k(
        self,
        ranked_items: np.ndarray,
        true_items: Set[int]
    ) -> float:
        """
        Compute Precision@K.

        Precision = |recommended ∩ relevant| / K
        """
        hits = len(set(ranked_items) & true_items)
        return hits / len(ranked_items) if len(ranked_items) > 0 else 0.0

    def _hit_at_k(
        self,
        ranked_items: np.ndarray,
        true_items: Set[int]
    ) -> float:
        """
        Compute Hit@K.

        Hit = 1 if any item in top-K is relevant, 0 otherwise
        """
        return 1.0 if len(set(ranked_items) & true_items) > 0 else 0.0

    def _mrr(
        self,
        ranked_items: np.ndarray,
        true_items: Set[int]
    ) -> float:
        """
        Compute MRR (Mean Reciprocal Rank).

        MRR = 1 / rank of first relevant item
        """
        for i, item in enumerate(ranked_items):
            if item in true_items:
                return 1.0 / (i + 1)
        return 0.0


def evaluate_model_on_dataset(
    model,
    edge_index: torch.Tensor,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    k_list: List[int] = [10, 20],
    batch_size: int = 256,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.

    Args:
        model: Recommendation model with predict() method
        edge_index: Graph edge indices
        test_df: Test DataFrame
        train_df: Train DataFrame (for excluding training items)
        user2idx: User to index mapping
        item2idx: Item to index mapping
        k_list: List of K values
        batch_size: Batch size for prediction
        device: Device to use

    Returns:
        Dictionary of metric values
    """
    model.eval()
    model = model.to(device)
    edge_index = edge_index.to(device)

    # Build ground truth and training data
    ground_truth = defaultdict(set)
    train_data = defaultdict(set)

    for _, row in test_df.iterrows():
        u_idx = user2idx.get(row['user_id'])
        i_idx = item2idx.get(row['movie_id'])
        if u_idx is not None and i_idx is not None:
            ground_truth[u_idx].add(i_idx)

    for _, row in train_df.iterrows():
        u_idx = user2idx.get(row['user_id'])
        i_idx = item2idx.get(row['movie_id'])
        if u_idx is not None and i_idx is not None:
            train_data[u_idx].add(i_idx)

    # Get predictions for all users
    predictions = {}
    test_users = list(ground_truth.keys())

    with torch.no_grad():
        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i:i+batch_size]
            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)

            # Predict scores for all items
            scores = model.predict(edge_index, batch_users_tensor, items=None)
            scores = scores.cpu().numpy()

            # Store predictions
            for j, user_idx in enumerate(batch_users):
                predictions[user_idx] = scores[j] if len(batch_users) > 1 else scores

    # Evaluate
    evaluator = RecommendationEvaluator(k_list=k_list)
    results = evaluator.evaluate(predictions, ground_truth, train_data)

    return results


def print_results(results: Dict[str, float]):
    """Pretty print evaluation results."""
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    # Group by metric type
    metrics_by_type = defaultdict(dict)
    for metric_name, value in results.items():
        if '@' in metric_name:
            metric_type, k = metric_name.split('@')
            metrics_by_type[metric_type][int(k)] = value
        else:
            metrics_by_type[metric_name][0] = value

    # Print each metric type
    for metric_type in ['NDCG', 'Recall', 'Precision', 'Hit', 'MRR']:
        if metric_type in metrics_by_type:
            values = metrics_by_type[metric_type]
            if 0 in values:
                # MRR (no K)
                print(f"{metric_type:12s}: {values[0]:.4f}")
            else:
                # Metrics with K
                k_values = sorted(values.keys())
                value_str = ', '.join([f"@{k}={values[k]:.4f}" for k in k_values])
                print(f"{metric_type:12s}: {value_str}")

    print("="*50 + "\n")
