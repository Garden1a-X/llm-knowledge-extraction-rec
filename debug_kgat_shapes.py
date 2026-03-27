#!/usr/bin/env python3
"""
Debug KGAT matrix shape issue
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from recbole.config import Config
from recbole.data import create_dataset

print("="*70)
print("DEBUG: KGAT Matrix Shapes")
print("="*70)

# Load config
config = Config(
    model='KGAT',
    dataset='ml-1m',
    config_file_list=['configs/recbole_kgat.yaml']
)

# Create dataset
print("\nCreating dataset...")
dataset = create_dataset(config)

print(f"\n✓ Dataset created successfully!")
print(f"  Users: {dataset.user_num}")
print(f"  Items: {dataset.item_num}")
print(f"  Entities: {dataset.entity_num}")
print(f"  Relations: {dataset.relation_num}")

# Check CKG graph
print(f"\nCKG Graph:")
ckg = dataset.ckg_graph(form="dgl", value_field="relation_id")
print(f"  Type: {type(ckg)}")
print(f"  Nodes: {ckg.num_nodes()}")
print(f"  Edges: {ckg.num_edges()}")

# Check adjacency matrix shape
print(f"\nTrying to get adjacency matrix...")
try:
    adj = ckg.adj_external(scipy_fmt="coo")
    print(f"  ✓ Adjacency matrix shape: {adj.shape}")
    print(f"  ✓ Expected shape: ({ckg.num_nodes()}, {ckg.num_nodes()})")

    if adj.shape[0] != ckg.num_nodes() or adj.shape[1] != ckg.num_nodes():
        print(f"  ⚠️  WARNING: Shape mismatch!")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
