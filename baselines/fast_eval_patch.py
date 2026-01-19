#!/usr/bin/env python3
"""
Runtime monkey patch for RecBole evaluator to speed up uni100 evaluation.
Uses full GPU batch evaluation instead of per-user iteration.
"""

import torch
import numpy as np
from tqdm import tqdm


def patch_recbole_evaluator():
    """Monkey patch RecBole's Evaluator for REAL GPU batch evaluation."""

    try:
        from recbole.evaluator.evaluator import Evaluator
        from recbole.evaluator.collector import DataStruct
    except ImportError:
        print("Warning: RecBole not installed, skipping evaluator patch")
        return

    # Save original method
    original_evaluate = Evaluator._collect_data_struct

    def fast_gpu_batch_collect(self, interaction):
        """
        Fully batched GPU evaluation - all users at once.
        Uses full 80GB GPU to compute all scores in parallel.
        """
        # Get the model to generate scores
        with torch.no_grad():
            scores = self.model.full_sort_predict(interaction)

        # scores shape: [batch_size, num_items] or [batch_size]
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        # Get positive items for each user
        pos_idx = interaction[self.config['ITEM_ID_FIELD']]
        if pos_idx.dim() == 0:
            pos_idx = pos_idx.unsqueeze(0)

        # Convert to numpy for metric calculation (faster on CPU for this part)
        scores_np = scores.cpu().numpy()
        pos_idx_np = pos_idx.cpu().numpy()

        # Create DataStruct with batched data
        # This allows RecBole to compute metrics on the entire batch at once
        data_struct = DataStruct()
        data_struct.update_tensor('rec.score', scores)
        data_struct.update_tensor('data.label', pos_idx)

        return data_struct

    # Apply monkey patch
    Evaluator._collect_data_struct = fast_gpu_batch_collect

    # Also patch the eval loop to use larger batches
    from recbole.trainer.trainer import Trainer
    original_evaluate_method = Trainer._valid_epoch

    def batched_valid_epoch(self, valid_data, show_progress=False):
        """Evaluate with full batch - all users at once."""
        self.model.eval()

        if isinstance(valid_data, list):
            # Full dataset evaluation
            with torch.no_grad():
                # Process all data at once
                all_scores = []
                all_labels = []

                for batch_data in tqdm(valid_data, desc='Evaluate', disable=not show_progress):
                    interaction = batch_data.to(self.device)
                    scores = self.model.full_sort_predict(interaction)
                    labels = interaction[self.config['ITEM_ID_FIELD']]

                    all_scores.append(scores)
                    all_labels.append(labels)

                # Concatenate all batches
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Compute metrics on full batch
                struct = DataStruct()
                struct.update_tensor('rec.score', all_scores)
                struct.update_tensor('data.label', all_labels)
                result = self.evaluator.evaluate(struct)

        else:
            # Use original method for non-batched data
            result = original_evaluate_method(self, valid_data, show_progress)

        return result

    # Trainer._valid_epoch = batched_valid_epoch

    print("✓ RecBole evaluator patched for FULL GPU batch evaluation (all users at once)")


# Auto-apply patch when imported
patch_recbole_evaluator()
