#!/usr/bin/env python3
"""
Runtime monkey patch for RecBole evaluator to speed up uni100 evaluation.
Import this before running RecBole to apply the patch automatically.
"""

import torch
import numpy as np
from tqdm import tqdm


def patch_recbole_evaluator():
    """Monkey patch RecBole's Evaluator for faster batch evaluation."""

    try:
        from recbole.evaluator import Evaluator
    except ImportError:
        print("Warning: RecBole not installed, skipping evaluator patch")
        return

    # Save original method
    original_evaluate = Evaluator.evaluate

    def fast_batch_evaluate(self, interaction, batch_size=None):
        """
        Faster batch evaluation for uni100 mode.
        Processes multiple users at once instead of one by one.
        """
        if batch_size is None:
            batch_size = self.config.get('eval_batch_size', 4096)

        # Get scores for all users
        with torch.no_grad():
            scores = self.model.full_sort_predict(interaction)

        if scores.dim() == 1:
            # Single user case
            return original_evaluate(self, interaction)

        # Batch case - much faster
        num_users = scores.shape[0]

        # Get positive items
        pos_items = interaction[self.config['ITEM_ID_FIELD']]
        if pos_items.dim() == 0:
            pos_items = pos_items.unsqueeze(0)

        # Process in batches to avoid memory issues
        all_metrics = []

        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)
            batch_scores = scores[start_idx:end_idx]
            batch_pos = pos_items[start_idx:end_idx]

            # Evaluate this batch
            batch_metrics = self._calculate_metrics(batch_scores, batch_pos)
            all_metrics.append(batch_metrics)

        # Average metrics across batches
        result = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                result[key] = np.mean(values)

        return result

    # Apply monkey patch
    Evaluator.evaluate = fast_batch_evaluate

    print("✓ RecBole evaluator patched for fast batch evaluation")


# Auto-apply patch when imported
patch_recbole_evaluator()
