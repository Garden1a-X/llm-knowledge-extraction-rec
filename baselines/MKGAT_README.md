# MKGAT Baseline Implementation

Implementation of **MKGAT (Multi-modal Knowledge Graph Attention Network)** for MovieLens 1M, based on the CIKM 2020 paper "Multi-modal Knowledge Graphs for Recommender Systems" by Rui Sun et al.

## Overview

MKGAT integrates visual features from movie posters with knowledge graph embeddings using attention-based aggregation. This implementation:

- **Visual Features**: ResNet50 (pretrained on ImageNet) extracts 2048-dim features from posters
- **Knowledge Graph**: Uses our existing LLM-extracted KG from `ml-1m.kg`
- **Model**: Attention-based neighbor aggregation with multi-modal fusion

## Quick Start

### Step 1: Extract Visual Features

Extract ResNet50 features from movie posters:

```bash
python baselines/extract_visual_features.py \
    --poster_dir /path/to/your/posters \
    --item_file data/recbole/ml-1m/ml-1m.item \
    --id_mapping data/recbole/ml-1m/id_mappings.json \
    --output data/recbole/ml-1m/visual_features.npy \
    --batch_size 32 \
    --device cuda
```

**Arguments:**
- `--poster_dir`: Directory containing poster images (named by original movie_id, e.g., `1.jpg`, `2.jpg`)
- `--item_file`: Path to `.item` file
- `--id_mapping`: Path to `id_mappings.json`
- `--output`: Where to save extracted features (`.npy` format)
- `--batch_size`: Batch size for extraction (default: 32)
- `--device`: `cuda` or `cpu`

**Output:**
- `visual_features.npy`: Numpy array of shape `(n_items, 2048)` with ResNet50 features

### Step 2: Run 5 Trials

Run MKGAT with 5 different random seeds for statistical significance:

```bash
bash baselines/run_mkgat_5trials.sh
```

This will:
1. Check that visual features exist
2. Run 5 trials with seeds: `[42, 2023, 2024, 2025, 12345]`
3. Save individual results for each trial
4. Aggregate results and compute mean ± std

**Output:**
```
outputs/mkgat_5trials/YYYYMMDD_HHMMSS/
├── seed_42_TIMESTAMP/
│   ├── results.json
│   └── best_model.pth
├── seed_2023_TIMESTAMP/
│   └── ...
├── ...
├── trial_1_seed_42.log
├── trial_2_seed_2023.log
├── ...
└── aggregated_results.json  # Mean ± Std across 5 trials
```

### Step 3: View Results

Aggregated results are automatically computed and saved to `aggregated_results.json`:

```json
{
  "n_trials": 5,
  "test_metrics": {
    "ndcg@10": {
      "mean": 0.1234,
      "std": 0.0056,
      "values": [0.1189, 0.1245, ...]
    },
    "recall@10": { ... },
    "precision@10": { ... }
  }
}
```

You can also manually aggregate results:

```bash
python baselines/aggregate_mkgat_results.py outputs/mkgat_5trials/YYYYMMDD_HHMMSS
```

## Manual Training (Single Seed)

For debugging or custom experiments, you can train with a single seed:

```bash
python baselines/train_mkgat.py \
    --data_dir data/recbole/ml-1m \
    --visual_features data/recbole/ml-1m/visual_features.npy \
    --output_dir outputs/mkgat/single_run \
    --embedding_dim 64 \
    --n_layers 3 \
    --aggregator_type bi-interaction \
    --dropout 0.1 \
    --reg_weight 1e-5 \
    --epochs 300 \
    --batch_size 1024 \
    --lr 0.001 \
    --early_stop 10 \
    --device cuda \
    --seed 42
```

**Key Arguments:**

**Model Architecture:**
- `--embedding_dim`: Embedding dimension (default: 64)
- `--n_layers`: Number of KG aggregation layers (default: 3)
- `--aggregator_type`: Aggregator type (`bi-interaction`, `gcn`, `graphsage`)
  - `bi-interaction`: Element-wise + feature-wise interactions (recommended, following KGAT paper)
  - `gcn`: GCN-style aggregation
  - `graphsage`: GraphSAGE-style concat
- `--dropout`: Dropout rate (default: 0.1)
- `--reg_weight`: L2 regularization weight (default: 1e-5)

**Training:**
- `--epochs`: Maximum epochs (default: 300)
- `--batch_size`: Training batch size (default: 1024)
- `--lr`: Learning rate (default: 0.001)
- `--early_stop`: Early stopping patience (default: 10 epochs)

**Other:**
- `--device`: `cuda` or `cpu`
- `--seed`: Random seed for reproducibility

## Model Architecture

### Visual Feature Extraction

```
Movie Poster (Image)
  ↓
ResNet50 (pretrained on ImageNet)
  ↓ Last hidden layer (avgpool)
2048-dim visual features
  ↓ MLP projection
64-dim visual embedding
```

### Multi-modal Fusion

```
Item Entity Embedding (64-dim)  +  Visual Embedding (64-dim)
  ↓ Concatenate
128-dim
  ↓ MLP fusion
64-dim Multi-modal Item Embedding
```

### KG Attention Aggregation

For each item, aggregate its neighbors through the KG:

```
Item/Entity
  ↓ Sample neighbors (relation, neighbor_entity)
Attention scores = <item_embed, relation_embed * neighbor_embed>
  ↓ Softmax
Weighted neighbor aggregation
  ↓ 3 layers (multi-hop)
Final enriched item embedding
```

### Prediction

```
User Embedding (64-dim)  ·  Item Embedding (64-dim)
  ↓ Dot product
Predicted score
```

## Implementation Details

### Data Format

**Input Files:**
- `ml-1m.inter`: User-item interactions with timestamps
- `ml-1m.kg`: Knowledge graph triplets (head, relation, tail)
- `ml-1m.item`: Item metadata
- `visual_features.npy`: Precomputed ResNet50 features

**KG Format:**
```
head_id:token    relation_id:token    tail_id:token
1                action_behaviors     dynamic_scenes
1                color_palette        colorful_tones
...
```

### Training Strategy

- **Data Split**: Temporal split (70% train / 10% val / 20% test)
- **Loss Function**: BPR (Bayesian Personalized Ranking) + L2 regularization
- **Negative Sampling**: Random negative items for each positive interaction
- **Early Stopping**: Based on validation NDCG@10, patience=10 epochs

### Evaluation Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Recall@10**: Proportion of relevant items retrieved
- **Precision@10**: Proportion of retrieved items that are relevant

## File Structure

```
baselines/
├── MKGAT_README.md                 # This file
├── extract_visual_features.py      # ResNet50 feature extraction
├── mkgat_model.py                  # MKGAT model implementation
├── train_mkgat.py                  # Training script
├── run_mkgat_5trials.sh            # Run 5 trials
└── aggregate_mkgat_results.py      # Aggregate trial results
```

## Notes on Data

**Poster Images:**
- The script expects posters named by **original MovieLens movie_id** (e.g., `1.jpg`, `2.jpg`)
- Uses `id_mappings.json` to map between RecBole IDs (1-indexed, continuous) and original movie IDs
- Missing posters will be represented as zero vectors (2048 zeros)

**Knowledge Graph:**
- Uses the existing LLM-extracted KG from Phase 4 extraction
- KG contains visual knowledge triplets like `(item, color_palette, warm_colors)`
- Entity IDs for non-item entities are hashed from entity names

**Why This Setup?**
- MKGAT paper doesn't provide code or datasets
- We use our own ML-1M dataset with:
  - ✅ Movie posters (for visual features)
  - ✅ User-item interactions
  - ✅ LLM-extracted knowledge graph
  - ❌ No text metadata (we focus on visual + KG only)

This will be clearly stated in the experiments section of our paper.

## Expected Performance

MKGAT is a strong multimodal baseline that:
- Uses CNN visual features (ResNet50)
- Integrates KG with attention mechanism
- Has been shown effective in prior work

We expect competitive performance compared to pure CF methods (BPR, LightGCN) and knowledge-based methods (KGAT).

## Comparison to Our Method

| Aspect | MKGAT (Baseline) | Our Method |
|--------|------------------|------------|
| **Visual Features** | CNN (ResNet50, 2048-dim) | LLM-extracted structured knowledge |
| **Interpretability** | ❌ Dense vectors | ✅ Structured triplets (relation, entity) |
| **Collaborative Filtering** | ⚠️ Limited (visual features are item-specific) | ✅ Strong (entities shared across items) |
| **Knowledge Graph** | Traditional KG + visual embeddings | LLM-extracted multimodal KG |

Our hypothesis: LLM-extracted knowledge provides **interpretable, shareable** entities that enable better collaborative filtering compared to dense CNN features.

## Troubleshooting

**"Visual features not found"**
- Run Step 1 (feature extraction) first
- Check that `--poster_dir` points to the correct location

**CUDA out of memory**
- Reduce `--batch_size` (try 512 or 256)
- Use `--device cpu` (slower but works)

**Poor performance**
- Check that visual features were extracted correctly
- Verify KG file is not empty
- Try different `--aggregator_type` (bi-interaction usually works best)

## Citation

If using MKGAT, please cite the original paper:

```bibtex
@inproceedings{sun2020mkgat,
  title={Multi-modal Knowledge Graphs for Recommender Systems},
  author={Sun, Rui and Cao, Xuezhi and Zhao, Yan and Wan, Junchen and Zhou, Kun and Zhang, Fuzheng and Wang, Zhongyuan and Zheng, Kai},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={1405--1414},
  year={2020}
}
```

---

**Good luck with your experiments!** 🚀
