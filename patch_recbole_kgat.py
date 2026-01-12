#!/usr/bin/env python3
"""
Patch RecBole KGAT to work with newer DGL versions.

The issue: RecBole 1.2.1 uses `preserve_nodes=True` in edge_subgraph(),
but newer DGL versions removed this parameter (it's now default behavior).

This script patches the KGAT model file to remove the problematic parameter.
"""

import os
import sys

def find_recbole_kgat():
    """Find RecBole KGAT model file."""
    try:
        import recbole
        recbole_path = os.path.dirname(recbole.__file__)
        kgat_path = os.path.join(
            recbole_path,
            'model',
            'knowledge_aware_recommender',
            'kgat.py'
        )

        if not os.path.exists(kgat_path):
            print(f"❌ KGAT file not found at: {kgat_path}")
            return None

        return kgat_path
    except ImportError:
        print("❌ RecBole not installed!")
        return None


def patch_kgat(kgat_path):
    """Patch KGAT to remove deprecated DGL parameters."""
    print(f"📝 Patching: {kgat_path}")

    # Read file
    with open(kgat_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup original (only if not already backed up)
    backup_path = kgat_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Backup created: {backup_path}")

    patches_applied = []
    patched_content = content

    # Patch 1: Remove preserve_nodes=True parameter
    # This line: dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True)
    # Becomes: dgl.edge_subgraph(self.ckg, edge_idxs)
    if 'preserve_nodes=True' in patched_content:
        patched_content = patched_content.replace(
            'dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True)',
            'dgl.edge_subgraph(self.ckg, edge_idxs)'
        )
        patches_applied.append("Removed preserve_nodes=True parameter")

    # Patch 2: Replace adjacency_matrix() with adj()
    # In newer DGL, adjacency_matrix() is removed, use adj() instead
    # Old: .adjacency_matrix(transpose=False, scipy_fmt="coo")
    # New: .adj(scipy_fmt="coo")

    if 'adjacency_matrix(transpose=False, scipy_fmt="coo")' in patched_content:
        patched_content = patched_content.replace(
            '.adjacency_matrix(transpose=False, scipy_fmt="coo")',
            '.adj(scipy_fmt="coo")'
        )
        patches_applied.append("Replaced adjacency_matrix() with adj()")

    if 'adjacency_matrix(transpose=True, scipy_fmt="coo")' in patched_content:
        patched_content = patched_content.replace(
            '.adjacency_matrix(transpose=True, scipy_fmt="coo")',
            '.adj(scipy_fmt="coo").T'
        )
        patches_applied.append("Replaced adjacency_matrix() with adj() and transposed result")

    # Also handle case where scipy_fmt is the only parameter
    if '.adjacency_matrix(scipy_fmt="coo")' in patched_content:
        patched_content = patched_content.replace(
            '.adjacency_matrix(scipy_fmt="coo")',
            '.adj(scipy_fmt="coo")'
        )
        patches_applied.append("Replaced adjacency_matrix(scipy_fmt) with adj(scipy_fmt)")

    if not patches_applied:
        print("✓ Already patched (or different version)")
        return True

    # Write patched file
    with open(kgat_path, 'w', encoding='utf-8') as f:
        f.write(patched_content)

    print("✓ Patches applied successfully!")
    for patch in patches_applied:
        print(f"  - {patch}")
    return True


def main():
    print("="*70)
    print("PATCH RECBOLE KGAT FOR DGL COMPATIBILITY")
    print("="*70)
    print()

    # Find KGAT file
    kgat_path = find_recbole_kgat()
    if not kgat_path:
        return 1

    print(f"Found KGAT at: {kgat_path}")
    print()

    # Patch
    if patch_kgat(kgat_path):
        print()
        print("="*70)
        print("✓ PATCH COMPLETE!")
        print("="*70)
        print()
        print("You can now run KGAT with:")
        print("  python scripts/run_recbole_baselines.py --model KGAT")
        print()
        print("To restore original:")
        print(f"  cp {kgat_path}.backup {kgat_path}")
        print("="*70)
        return 0
    else:
        print()
        print("❌ Patch failed - manual intervention needed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
