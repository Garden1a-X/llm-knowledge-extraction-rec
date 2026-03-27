# MMGCN and VBPR Implementation Plan

## Overview

Implementing two visual-enhanced recommendation baselines for MovieLens-1M:
1. **VBPR** - Visual Bayesian Personalized Ranking (AAAI 2016)
2. **MMGCN** - Multi-modal Graph Convolution Network (ACM MM 2019)

Both methods use **ResNet50 visual features** extracted from movie posters (same as MKGAT).

---

## 1. VBPR (Visual Bayesian Personalized Ranking)

### Paper
He & McAuley. "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback." AAAI 2016.

### Core Idea
Extend BPR (Bayesian Personalized Ranking) by incorporating visual features from item images.

### Architecture

```
User u, Item i, Visual Features v (2048-dim from ResNet50)

Embeddings:
- u_embed: (n_users, embedding_dim) - user latent factors
- i_embed: (n_items, embedding_dim) - item latent factors
- E: (embedding_dim, visual_dim) - visual embedding matrix
- v: (visual_dim,) - visual features (ResNet50)

Prediction:
score(u, i) = u_embed · i_embed + u_embed · (E · v_i)
            = [CF component] + [Visual component]

Where:
- u_embed · i_embed: Standard collaborative filtering (like BPR)
- u_embed · (E · v_i): User-specific visual preference
  - E · v_i: Projects visual features to embedding space
  - u_embed · (...): User's affinity to these visual features
```

### Key Components

1. **User Embeddings**: `(n_users, embedding_dim)`
2. **Item Embeddings**: `(n_items, embedding_dim)`
3. **Visual Embedding Matrix E**: `(embedding_dim, 2048)` - projects ResNet features
4. **Visual Features**: `(n_items, 2048)` - precomputed ResNet50 features

### Loss Function

BPR loss with regularization:
```
Loss = -log(σ(score_pos - score_neg)) + λ(||u||² + ||i_pos||² + ||i_neg||² + ||E||²)
```

### Implementation Notes

- Uses BPR sampling (1 positive + 1 negative per user)
- Visual features are precomputed and frozen
- Only E matrix is learned for visual processing
- Simple but effective: directly models user-visual preference

---

## 2. MMGCN (Multi-modal Graph Convolution Network)

### Paper
Wei et al. "MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video." ACM MM 2019.

### Core Idea
Build **modality-specific user-item graphs** and use GCN to propagate information, then fuse with attention.

### Architecture (Visual-only version)

Since we only have visual features for MovieLens-1M, we use:
- **Visual Graph**: User-Item graph with visual features
- **CF Graph**: User-Item graph with standard embeddings

```
Input:
- User-Item interaction graph G
- Visual features V (n_items, 2048) from ResNet50

For each modality m ∈ {visual, CF}:
  1. Initialize embeddings: u_m^(0), i_m^(0)
  2. GCN propagation (L layers):
     u_m^(l+1) = Aggregate(neighbors of u in G, weighted by i_m^(l))
     i_m^(l+1) = Aggregate(neighbors of i in G, weighted by u_m^(l))
  3. Combine layers: u_m = CONCAT(u_m^(0), ..., u_m^(L))

Modal Attention:
  α_m = softmax(w_m · u_m)  # Learn importance of each modality
  u_final = Σ α_m · u_m

Prediction:
  score(u, i) = u_final · i_final
```

### Key Components

1. **Modality-specific embeddings**:
   - CF embeddings: `(n_users/n_items, embedding_dim)` - learned
   - Visual embeddings: `(n_items, embedding_dim)` - projected from ResNet features

2. **GCN Layers** (per modality):
   - Message passing on user-item bipartite graph
   - Similar to LightGCN but separate for each modality

3. **Modal Attention**:
   - Learn attention weights for each modality
   - Adaptively fuse CF and visual information

4. **Layer Combination**:
   - Concatenate embeddings from all layers (like LightGCN)
   - Or weighted sum

### Implementation Details

**Graph Construction**:
- User-Item bipartite graph from interactions
- Normalized adjacency matrix: D^(-1/2) A D^(-1/2)

**Visual Feature Processing**:
```python
# Project ResNet features to embedding space
visual_proj = nn.Linear(2048, embedding_dim)
visual_embed = visual_proj(resnet_features)  # (n_items, embedding_dim)
```

**GCN Propagation** (per modality):
```python
for layer in range(n_layers):
    # User aggregation
    u_embed_new = aggregate_from_items(u_embed, i_embed, adj_matrix)

    # Item aggregation
    i_embed_new = aggregate_from_users(i_embed, u_embed, adj_matrix)

    u_embed, i_embed = u_embed_new, i_embed_new
```

**Modal Attention**:
```python
# Learn attention weights
alpha_cf = attention_weight(u_cf_embed)
alpha_visual = attention_weight(u_visual_embed)

# Softmax normalization
alpha = softmax([alpha_cf, alpha_visual])

# Fused representation
u_final = alpha[0] * u_cf_embed + alpha[1] * u_visual_embed
```

---

## Comparison: VBPR vs MMGCN

| Aspect | VBPR | MMGCN |
|--------|------|-------|
| **Graph** | ❌ No graph structure | ✅ GCN on user-item graph |
| **Visual Integration** | Direct: u·(E·v) | GCN propagation with visual features |
| **Complexity** | Simple MF + visual | Complex GCN + modal attention |
| **User-Visual Modeling** | Linear interaction | Graph-based propagation |
| **Expected Performance** | Baseline | Stronger (graph + attention) |

---

## Implementation Plan

### Phase 1: VBPR (Simpler, ~2-3 hours)

**Files**:
- `baselines/vbpr_model.py` - VBPR model
- `baselines/train_vbpr.py` - Training script with uni100 evaluation
- `baselines/run_vbpr_5trials.sh` - 5-trial runner

**Steps**:
1. Implement VBPR model (user/item embeddings + visual projection)
2. BPR loss with negative sampling
3. Training loop with early stopping
4. uni100 evaluation (1 positive + 99 random negatives)
5. 5 trials with seeds [42, 2023, 2024, 2025, 12345]

### Phase 2: MMGCN (More complex, ~4-5 hours)

**Files**:
- `baselines/mmgcn_model.py` - MMGCN model
- `baselines/train_mmgcn.py` - Training script
- `baselines/run_mmgcn_5trials.sh` - 5-trial runner

**Steps**:
1. Build user-item adjacency matrix
2. Implement modality-specific GCN layers
3. Implement modal attention mechanism
4. BPR loss + training loop
5. uni100 evaluation
6. 5 trials

---

## Shared Components

Both methods use:
- **Visual features**: `data/recbole/ml-1m/visual_features.npy` (already extracted)
- **Dataset**: MovieLens-1M with 5-core filtering
- **Evaluation**: uni100 mode (1 pos + 99 neg)
- **Metrics**: NDCG@10, Recall@10, Precision@10
- **Trials**: 5 runs with seeds [42, 2023, 2024, 2025, 12345]

---

## Expected Timeline

- **VBPR**: 2-3 hours
- **MMGCN**: 4-5 hours
- **Testing + Debugging**: 1-2 hours
- **Total**: ~8-10 hours

---

## Key References

1. **VBPR**: He, R., & McAuley, J. (2016). VBPR: Visual bayesian personalized ranking from implicit feedback. In AAAI.

2. **MMGCN**: Wei, Y., Wang, X., Nie, L., He, X., Hong, R., & Chua, T. S. (2019). MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video. In ACM MM.

---

*Created: 2026-01-21*
