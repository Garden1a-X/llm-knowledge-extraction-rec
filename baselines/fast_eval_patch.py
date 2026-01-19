#!/usr/bin/env python3
"""
Runtime monkey patch for RecBole to use large eval batches.
Simply disables the patch that was causing issues.
The large eval_batch_size in config (100000) will handle it.
"""

def patch_recbole_evaluator():
    """No-op patch - just rely on eval_batch_size config."""
    print("✓ Using eval_batch_size from config (no patch needed)")

# Auto-apply when imported
patch_recbole_evaluator()
