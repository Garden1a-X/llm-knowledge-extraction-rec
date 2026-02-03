# ML-1M Ablation Experiment Results

## Experimental Setup
- Evaluation: uni100 (1 positive + 99 negatives per user)
- 5 trials with seeds: [42, 123, 456, 789, 2024]
- Data split: Per-user temporal split (7:1:2)

## Ablation Results

### Knowledge Graph Component Analysis

| Method | NDCG@10 | Recall@10 | Description |
|--------|---------|-----------|-------------|
| **Ours (Full)** | TBD | TBD | Full model with both Item KG and User KG |
| Ours (Item KG only) | 0.2637 ± 0.0045 | 0.4898 ± 0.0076 | Only Item KG, no User KG |
| Ours (User KG only) | 0.2699 ± 0.0051 | 0.4997 ± 0.0090 | Only User KG, no Item KG |

### User KG Interest Type Analysis

| Method | NDCG@10 | Recall@10 | Description |
|--------|---------|-----------|-------------|
| **Ours (Full)** | TBD | TBD | Full User KG (both long-term and short-term) |
| w/o Long-term | 0.2696 ± 0.0032 | 0.4996 ± 0.0055 | User KG without long-term interests |
| w/o Short-term | 0.2689 ± 0.0007 | 0.4979 ± 0.0014 | User KG without short-term interests |

## Key Observations

1. **Item KG vs User KG**:
   - User KG only (0.2699) > Item KG only (0.2637) in NDCG@10
   - User KG provides +0.0062 NDCG@10 improvement over Item KG alone
   - This suggests user-side knowledge is more valuable for recommendation

2. **Long-term vs Short-term Interests**:
   - w/o Long-term (0.2696) ≈ w/o Short-term (0.2689)
   - Both interest types contribute similarly to model performance
   - Removing either type causes similar degradation

3. **Variance Analysis**:
   - w/o Short-term has lowest variance (±0.0007), suggesting more stable predictions
   - Item KG only has higher variance (±0.0045), indicating less robust

## Raw Results

### Item KG Only (5 trials)
```
NDCG@10: 0.2637 ± 0.0045
Recall@10: 0.4898 ± 0.0076
```

### User KG Only (5 trials)
```
NDCG@10: 0.2699 ± 0.0051
Recall@10: 0.4997 ± 0.0090
```

### w/o Long-term (5 trials)
```
NDCG@10: 0.2696 ± 0.0032
Recall@10: 0.4996 ± 0.0055
```

### w/o Short-term (5 trials)
```
NDCG@10: 0.2689 ± 0.0007
Recall@10: 0.4979 ± 0.0014
```

---
*Results recorded: 2026-02-03*
