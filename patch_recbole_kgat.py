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
    """Patch KGAT to remove preserve_nodes parameter."""
    print(f"📝 Patching: {kgat_path}")

    # Read file
    with open(kgat_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if 'preserve_nodes=True' not in content:
        print("✓ Already patched (or different version)")
        return True

    # Backup original
    backup_path = kgat_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Backup created: {backup_path}")

    # Apply patch: remove preserve_nodes=True parameter
    # This line: dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True)
    # Becomes: dgl.edge_subgraph(self.ckg, edge_idxs)

    patched_content = content.replace(
        'dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True)',
        'dgl.edge_subgraph(self.ckg, edge_idxs)'
    )

    if patched_content == content:
        print("⚠️  Pattern not found - RecBole version may be different")
        print("    Looking for: dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True)")
        return False

    # Write patched file
    with open(kgat_path, 'w', encoding='utf-8') as f:
        f.write(patched_content)

    print("✓ Patch applied successfully!")
    print(f"  Changed: preserve_nodes=True -> removed")
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
