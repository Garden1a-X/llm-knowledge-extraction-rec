# Method Section Technical Specification

**Purpose**: This document provides detailed technical information about the model architecture based on actual code implementation, for writing the paper's Method section.

**Generated**: 2026-01-20
**Code Base**: LLM Knowledge Extraction for Recommendation

---

## Summary

This document answers detailed technical questions about your model implementation to help you write the Method section of your paper. All information is extracted directly from the actual code.

**Key Points**:
1. All embeddings use Xavier Uniform initialization
2. Entity masks use frequency-based logit initialization
3. Heterogeneous graph has 7 edge types (3 forward + 3 reverse + 1 rating)
4. CF view: Multi-layer GAT on user-item bipartite graph
5. KG view: HeteroConv with 6 GAT sub-layers
6. 4 loss components: L_rec (InfoNCE) + L_contrast (multi-view) + L_align (entity-item) + L_mask (regularization)
7. 2-layer MLP fusion for combining CF and KG views
8. Per-user temporal data split with training-only graph construction
9. Two evaluation modes: Full ranking and Uni100

---

## Category 1: Embeddings Initialization

### Q1: How are user, item, and entity embeddings initialized?

**Answer**: All embeddings use **Xavier Uniform initialization**.

**Code Reference**: `src/model/ours.py:56-61`

```python
self.user_embed = nn.Embedding(num_users, embedding_dim)
self.item_embed = nn.Embedding(num_items, embedding_dim)
self.entity_embed = nn.Embedding(num_entities, embedding_dim)

nn.init.xavier_uniform_(self.user_embed.weight)
nn.init.xavier_uniform_(self.item_embed.weight)
nn.init.xavier_uniform_(self.entity_embed.weight)
```

**Design Principle**: Xavier initialization helps maintain gradient variance across layers, preventing vanishing/exploding gradients in deep GAT networks.

**Mathematical Formulation**:
```
W ~ U[-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))]
where fan_in = fan_out = 1 for embeddings
```

---

## Category 2: Entity Mask Design

### Q2: How is the learnable entity mask initialized and designed?

**Answer**: The mask uses **frequency-based logit initialization** with continuous sigmoid activation.

**Architecture** (`src/model/ours.py:63-70`):
```python
if mask_init is not None:
    # Frequency-based initialization
    eps = 1e-7
    mask_init_clamped = torch.clamp(mask_init, eps, 1 - eps)
    self.mask_logits = nn.Parameter(torch.logit(mask_init_clamped))
else:
    # All-ones initialization (no masking)
    self.mask_logits = nn.Parameter(torch.zeros(num_entities))
```

**Mask Application** (`src/model/ours.py:120-122`):
```python
def get_mask(self):
    """Get continuous mask in [0, 1]"""
    return torch.sigmoid(self.mask_logits)
```

**Frequency-Based Initialization Strategy** (`src/data/graph_builder.py:298-339`):

The initial mask values are computed based on entity frequency in the KG:

```python
def compute_frequency_mask(entity_freq, entity_id_map, min_freq=5, max_freq=1000):
    # Low-frequency entities (<min_freq): mask=0.3 (potentially unreliable/hallucinated)
    # Normal-frequency entities: mask=1.0 (trustworthy)
    # High-frequency entities (>max_freq): mask=0.8 (potentially too generic)
```

**Design Principles**:
1. **Continuous masks** (not binary): Allow gradient flow and fine-grained control
2. **Logit space learning**: Model learns unbounded logits, sigmoid ensures [0,1] output
3. **Frequency-based prior**: Incorporates domain knowledge about entity reliability
4. **Per-entity learnable**: Each entity has independent mask weight

**Mathematical Formulation**:
```
mask_logits ∈ R^{|E|}  (learnable parameters)
mask = σ(mask_logits) ∈ [0,1]^{|E|}
masked_entity_emb = mask ⊙ entity_emb
```

---

## Category 3: Graph Structure Design

### Q3: What is the complete edge type structure in the heterogeneous graph?

**Answer**: The heterogeneous graph contains **7 edge types** (3 forward + 3 reverse + 1 rating).

**Edge Types** (`src/data/graph_builder.py:208-271`):

| Edge Type | Direction | Source | Description | Code Ref |
|-----------|-----------|--------|-------------|----------|
| `long_term` | Forward | User → Entity | Long-term interest (84-day summary) | `graph_builder.py:221` |
| `short_term` | Forward | User → Entity | Short-term interest (21-day buckets) | `graph_builder.py:235` |
| `describes` | Forward | Entity → Item | Entity describes item attributes | `graph_builder.py:251` |
| `rev_long_term` | Reverse | Entity → User | Reverse of long_term | `graph_builder.py:225` |
| `rev_short_term` | Reverse | Entity → User | Reverse of short_term | `graph_builder.py:239` |
| `rev_describes` | Reverse | Item → Entity | Reverse of describes | `graph_builder.py:255` |
| `rated` | Rating | User → Item | User-item interactions (for CF view) | `graph_builder.py:268` |

**Graph Construction Code**:
```python
# User-Entity edges (bidirectional)
graph['user', 'long_term', 'entity'].edge_index = ...
graph['entity', 'rev_long_term', 'user'].edge_index = ...

graph['user', 'short_term', 'entity'].edge_index = ...
graph['entity', 'rev_short_term', 'user'].edge_index = ...

# Entity-Item edges (bidirectional)
graph['entity', 'describes', 'item'].edge_index = ...
graph['item', 'rev_describes', 'entity'].edge_index = ...

# User-Item edges (for CF view)
graph['user', 'rated', 'item'].edge_index = ...
```

**Design Principle**: Bidirectional edges enable message passing in both directions, allowing entities to aggregate information from both users and items.

### Q4: Are edge weights used? How?

**Answer**: **No learnable edge weights**. Only the `rated` edge stores rating values as attributes (not used in GNN message passing).

**Code Evidence** (`src/data/graph_builder.py:259-269`):
```python
def _add_user_item_edges(self, graph: HeteroData, inter: pd.DataFrame):
    """Add User-Item edges (rating interactions)"""
    # ...
    edge_index = torch.tensor([user_ids, item_ids], dtype=torch.long)
    edge_attr = torch.tensor(ratings, dtype=torch.float)  # Stored but not used

    graph['user', 'rated', 'item'].edge_index = edge_index
    graph['user', 'rated', 'item'].edge_attr = edge_attr  # Only for reference
```

**Design Principle**:
- GAT already learns attention weights for message aggregation
- Explicit edge weights would add unnecessary complexity
- Rating values are stored for reference but not used in GNN layers

---

## Category 4: GNN Architecture - CF View

### Q5: How is the CF (Collaborative Filtering) view encoder designed?

**Answer**: CF view uses a **multi-layer GAT on user-item bipartite graph**.

**Architecture** (`src/model/encoders.py:14-78`):

```python
class CFEncoder(nn.Module):
    """CF view: User-Item bipartite graph with GAT"""

    def __init__(self, num_users, num_items, embedding_dim=64,
                 num_layers=2, gat_heads=4, dropout=0.2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        # Multi-layer GAT
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = GATConv(
                embedding_dim,
                embedding_dim // gat_heads,  # Per-head dimension
                heads=gat_heads,
                dropout=dropout,
                concat=True  # Concatenate heads
            )
            self.convs.append(conv)
```

**Forward Pass**:
```python
def forward(self, x, edge_index):
    """
    Args:
        x: [num_users + num_items, embedding_dim] - Concatenated embeddings
        edge_index: [2, num_edges] - Bipartite graph edges

    Returns:
        user_emb: [num_users, embedding_dim]
        item_emb: [num_items, embedding_dim]
    """
    h = x
    for i, conv in enumerate(self.convs):
        h = conv(h, edge_index)

        # Apply ReLU + Dropout (except last layer)
        if i < len(self.convs) - 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

    # Split back into user and item embeddings
    user_emb = h[:self.num_users]
    item_emb = h[self.num_users:]

    return user_emb, item_emb
```

**Graph Construction for CF View** (`src/data/graph_builder.py:273-295`):
```python
def _build_cf_graph(self, inter: pd.DataFrame) -> torch.Tensor:
    """Build CF graph (User-Item bipartite graph)"""
    user_ids = [self.user_id_map[uid] for uid in inter['user_id:token']]

    # Item IDs need offset (items come after users in node ordering)
    item_ids = [
        self.item_id_map[iid] + len(self.user_id_map)
        for iid in inter['item_id:token']
    ]

    # Bidirectional edges (User→Item and Item→User)
    edge_index = torch.tensor(
        [user_ids + item_ids, item_ids + user_ids],
        dtype=torch.long
    )
    return edge_index
```

**Design Principles**:
1. **Bipartite structure**: Users and items in same node space (users: 0 to N-1, items: N to N+M-1)
2. **Bidirectional edges**: Message passing in both directions
3. **Multi-head attention**: GAT learns importance of neighbors
4. **Layer-wise activation**: ReLU after each layer except last

---

## Category 5: GNN Architecture - KG View

### Q6: How is the KG (Knowledge Graph) view encoder designed?

**Answer**: KG view uses **HeteroConv with 6 GAT sub-layers** (one for each edge type).

**Architecture** (`src/model/encoders.py:81-158`):

```python
class KGEncoder(nn.Module):
    """KG view: User-Entity-Item heterogeneous graph with HeteroConv + GAT"""

    def __init__(self, embedding_dim=64, num_layers=2, gat_heads=4, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # HeteroConv: different GNN for each edge type
            hetero_conv = HeteroConv({
                ('user', 'long_term', 'entity'): GATConv(...),
                ('user', 'short_term', 'entity'): GATConv(...),
                ('entity', 'describes', 'item'): GATConv(...),
                ('entity', 'rev_long_term', 'user'): GATConv(...),
                ('entity', 'rev_short_term', 'user'): GATConv(...),
                ('item', 'rev_describes', 'entity'): GATConv(...),
            }, aggr='sum')  # Aggregate multiple edge types via summation

            self.convs.append(hetero_conv)
```

**Forward Pass**:
```python
def forward(self, x_dict, edge_index_dict):
    """
    Args:
        x_dict: {'user': [N_u, d], 'entity': [N_e, d], 'item': [N_i, d]}
        edge_index_dict: {edge_type: [2, num_edges]}

    Returns:
        x_dict: Updated node embeddings
    """
    for i, conv in enumerate(self.convs):
        # HeteroConv handles all edge types simultaneously
        x_dict = conv(x_dict, edge_index_dict)

        # Apply ReLU + Dropout to all node types (except last layer)
        if i < len(self.convs) - 1:
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                     for key, x in x_dict.items()}

    return x_dict
```

**Design Principles**:
1. **Type-specific message passing**: Different GAT for each edge type
2. **HeteroConv aggregation**: Sum messages from different edge types
3. **Reverse edges included**: Bidirectional information flow
4. **Shared layer-wise processing**: Same activation/dropout for all node types

---

## Category 6: GAT Details

### Q7: What are the specific GAT hyperparameters?

**Answer**: GAT configuration from `configs/ours_full_videogames.yaml` and code:

| Parameter | Value | Description | Code Reference |
|-----------|-------|-------------|----------------|
| `embedding_dim` | 64 | Node embedding dimension | `ours_full_videogames.yaml:16` |
| `num_gnn_layers` | 2 | Number of GAT layers | `ours_full_videogames.yaml:17` |
| `gat_heads` | 4 | Number of attention heads | `ours_full_videogames.yaml:18` |
| `dropout` | 0.2 | Dropout rate | `ours_full_videogames.yaml:19` |
| Per-head dim | 16 | embedding_dim // gat_heads | `encoders.py:31` |
| Head concat | True | Concatenate attention heads | `encoders.py:34` |
| Attention dropout | 0.2 | Dropout on attention weights | `encoders.py:33` |

**GAT Layer Construction**:
```python
GATConv(
    in_channels=64,           # Input dimension
    out_channels=16,          # Per-head output (64/4)
    heads=4,                  # Number of attention heads
    dropout=0.2,              # Attention dropout
    concat=True               # Concatenate heads → output: 16*4=64
)
```

**Mathematical Formulation**:
```
h_i^{(l+1)} = ||_{k=1}^K σ(∑_{j∈N(i)} α_{ij}^k W^k h_j^{(l)})

where:
- K = 4 (number of heads)
- α_{ij}^k = attention weight (learned via GAT)
- W^k ∈ R^{64×16} (per-head weight matrix)
- || denotes concatenation
- σ = ReLU (except last layer)
```

---

## Category 7: Loss Function Design

### Q8-Q11: Loss Components

**Total Loss** (`src/model/losses.py:244-267`):
```python
L_total = L_rec + α·L_contrast + β·L_align + γ·L_mask

where:
α = 0.1 (multi-view contrastive weight)
β = 0.05 (entity-item alignment weight)
γ = 0.01 (mask regularization weight)
```

**L_rec (Recommendation Loss)** - InfoNCE:
```
L_rec = -log(exp(u·i_pos / τ) / (exp(u·i_pos / τ) + ∑_{i_neg} exp(u·i_neg / τ)))
τ = 0.2
```

**L_contrast (Multi-view Contrastive)** - CF vs KG views:
```
L_contrast = -log(exp(u_cf · u_kg / τ) / (exp(u_cf · u_kg / τ) + ∑_{u'∈B} exp(u_cf · u'_kg / τ)))
τ = 0.1 (normalized embeddings)
```

**L_align (Entity-Item Alignment)**:
```
L_align = -log(exp(e·i_pos / τ) / (exp(e·i_pos / τ) + ∑_{i_neg} exp(e·i_neg / τ)))
τ = 0.2, num_neg = 5
```

**L_mask (Mask Regularization)**:
```
L_mask = λ_sparse · ∑_e (1 - m_e) - λ_entropy · (1/|E|) ∑_e H(m_e)
λ_sparse = 1.0, λ_entropy = 0.1
```

---

## Category 8: Training Details

**Batch Construction** (`src/data/dataset.py:86-114`):
- Each sample: 1 positive + 1 negative item
- Negatives sampled from items user hasn't interacted with
- Batch size: 2048

**MLP Fusion** (`src/model/ours.py:87-115`):
```python
# 2-layer MLP
fusion = nn.Sequential(
    nn.Linear(128, 64),  # embedding_dim * 2 → embedding_dim
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 64)
)
```

**Data Split** (`src/data/dataset.py:142-225`):
- Per-user temporal split: 70% train, 10% val, 20% test
- Training-only graph construction (no data leakage)
- Min 3 interactions per user required

**Inference** (`src/utils/metrics.py:199-332`):
- Two modes: Full ranking or Uni100 (1 pos + 99 neg)
- Inner product scoring: score(u, i) = u^T · i
- Metrics: NDCG@K, Recall@K, Precision@K, Hit@K

---

## Complete Hyperparameter Table

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | embedding_dim | 64 |
| | num_gnn_layers | 2 |
| | gat_heads | 4 |
| | dropout | 0.2 |
| **Loss** | alpha_contrast | 0.1 |
| | beta_align | 0.05 |
| | gamma_mask | 0.01 |
| | temperature_rec | 0.2 |
| | temperature_contrast | 0.1 |
| **Training** | batch_size | 2048 |
| | learning_rate | 0.001 |
| | num_epochs | 300 |
| | early_stop_patience | 10 |
| | eval_mode | uni100 |
| | random_seeds | 42, 2023, 2024, 2025, 12345 |

---

**End of Document**
