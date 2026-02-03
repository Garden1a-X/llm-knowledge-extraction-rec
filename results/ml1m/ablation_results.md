# ML-1M Ablation Experiment Results

## Experimental Setup
- Evaluation: uni100 (1 positive + 99 negatives per user)
- 5 trials with seeds: [42, 123, 456, 789, 2024]
- Data split: Per-user temporal split (7:1:2)

## Ablation Results

### Knowledge Graph Component Analysis

| Method | NDCG@10 | Recall@10 | Description |
|--------|---------|-----------|-------------|
| **Ours (Full)** | **0.2707 ± 0.0026** | **0.5006 ± 0.0041** | Full model with both Item KG and User KG |
| Ours (Item KG only) | 0.2637 ± 0.0045 | 0.4898 ± 0.0076 | Only Item KG, no User KG |
| Ours (User KG only) | 0.2699 ± 0.0051 | 0.4997 ± 0.0090 | Only User KG, no Item KG |

### User KG Interest Type Analysis

| Method | NDCG@10 | Recall@10 | Description |
|--------|---------|-----------|-------------|
| **Ours (Full)** | **0.2707 ± 0.0026** | **0.5006 ± 0.0041** | Full User KG (both long-term and short-term) |
| w/o Long-term | 0.2696 ± 0.0032 | 0.4996 ± 0.0055 | User KG without long-term interests |
| w/o Short-term | 0.2689 ± 0.0007 | 0.4979 ± 0.0014 | User KG without short-term interests |

## Key Observations

1. **Item KG vs User KG**:
   - Full (0.2707) > User KG only (0.2699) > Item KG only (0.2637)
   - **User KG more important**: User KG only achieves 99.7% of Full performance
   - **Item KG alone drops 2.6%**: Item KG only loses -0.0070 NDCG@10 vs Full
   - **Both KGs are complementary**: Full model outperforms either KG alone

2. **Long-term vs Short-term Interests**:
   - Full (0.2707) > w/o Long-term (0.2696) > w/o Short-term (0.2689)
   - **Short-term interests slightly more valuable**: removing short-term causes -0.0018 vs removing long-term -0.0011
   - Both interest types contribute to model performance

3. **Relative Importance (% drop from Full)**:
   - Item KG only: -2.6% (most significant)
   - User KG only: -0.3%
   - w/o Long-term: -0.4%
   - w/o Short-term: -0.7%

## Summary Table

| Method | NDCG@10 | Recall@10 | Δ NDCG@10 |
|--------|---------|-----------|-----------|
| **Ours (Full)** | **0.2707 ± 0.0026** | **0.5006 ± 0.0041** | - |
| Item KG only | 0.2637 ± 0.0045 | 0.4898 ± 0.0076 | -2.6% |
| User KG only | 0.2699 ± 0.0051 | 0.4997 ± 0.0090 | -0.3% |
| w/o Long-term | 0.2696 ± 0.0032 | 0.4996 ± 0.0055 | -0.4% |
| w/o Short-term | 0.2689 ± 0.0007 | 0.4979 ± 0.0014 | -0.7% |

---
*Results recorded: 2026-02-03*
