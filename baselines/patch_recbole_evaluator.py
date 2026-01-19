#!/usr/bin/env python3
"""
Patch RecBole evaluator to support batch evaluation for uni100 mode.
This significantly speeds up evaluation by processing multiple users in parallel.
"""

import os
import sys
from pathlib import Path

def find_recbole_evaluator():
    """Find RecBole evaluator.py file."""
    try:
        import recbole.evaluator.evaluator as evaluator_module
        import inspect
        return Path(inspect.getfile(evaluator_module))
    except ImportError:
        print("Error: RecBole not found in Python path")
        print("Make sure you're running this in the correct conda environment")
        return None

def patch_evaluator(evaluator_path):
    """Patch the evaluator to use batch evaluation."""

    print(f"Found evaluator at: {evaluator_path}")

    # Read the original file
    with open(evaluator_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if 'PATCHED FOR BATCH EVAL' in content:
        print("✓ Evaluator already patched!")
        return True

    # Backup original file
    backup_path = evaluator_path.with_suffix('.py.backup')
    if not backup_path.exists():
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created backup: {backup_path}")

    # Find the _test_by_mask method and replace it
    # This is the method that does the actual evaluation

    patch_code = '''
    def _test_by_mask(self, batch_size=None):
        """Evaluate with batch processing for speed - PATCHED FOR BATCH EVAL"""
        interaction = self.tot_item_num
        scores = self.model.full_sort_predict(interaction)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf  # Mask padding item

        if self.config['eval_type'] == EvaluatorType.RANKING:
            if batch_size is None:
                batch_size = self.config.get('eval_batch_size', 4096)

            # Batch evaluation for uni100 mode
            pos_items = interaction[self.config['ITEM_ID_FIELD']].view(-1, 1)

            # Process in batches
            num_users = scores.shape[0]
            all_metrics = []

            for start_idx in range(0, num_users, batch_size):
                end_idx = min(start_idx + batch_size, num_users)
                batch_scores = scores[start_idx:end_idx]
                batch_pos = pos_items[start_idx:end_idx]

                # Compute metrics for this batch
                batch_result = self.evaluator.evaluate(batch_scores, batch_pos)
                all_metrics.append(batch_result)

            # Aggregate results from all batches
            result = {}
            for metric in all_metrics[0].keys():
                values = [m[metric] for m in all_metrics]
                result[metric] = np.mean(values)

            return result
        else:
            return self._calculate_metrics(scores, pos_items)
'''

    # Try to find and replace the method
    import re

    # Pattern to find the _test_by_mask method
    pattern = r'(\s+)def _test_by_mask\(self[^)]*\):.*?(?=\n\s+def |\nclass |\Z)'

    if not re.search(pattern, content, re.DOTALL):
        print("Warning: Could not find _test_by_mask method to patch")
        print("RecBole version may have changed. Manual patching required.")
        return False

    # Replace with patched version
    new_content = re.sub(
        pattern,
        patch_code,
        content,
        flags=re.DOTALL
    )

    # Write patched file
    with open(evaluator_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("✓ Successfully patched evaluator!")
    print("  Evaluation will now use batch processing")
    print(f"  Backup saved to: {backup_path}")

    return True

def main():
    print("="*70)
    print("RecBole Evaluator Batch Evaluation Patch")
    print("="*70)
    print()

    evaluator_path = find_recbole_evaluator()

    if evaluator_path is None:
        print("\nMake sure you activate the correct conda environment first:")
        print("  conda activate xuao_llm_kg_rec")
        return 1

    if not evaluator_path.exists():
        print(f"Error: Evaluator file not found: {evaluator_path}")
        return 1

    success = patch_evaluator(evaluator_path)

    if success:
        print("\n✓ Patch complete! Evaluation should now be much faster.")
        print("\nTo restore original evaluator:")
        print(f"  cp {evaluator_path}.backup {evaluator_path}")
        return 0
    else:
        print("\n✗ Patch failed!")
        print("\nYou may need to manually edit the RecBole evaluator.")
        print(f"File location: {evaluator_path}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
