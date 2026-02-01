# Amazon Beauty 实验结果汇总

## 实验设置
- 数据集: Amazon Beauty (无5-core过滤)
- 评估模式: uni100 (1 positive + 99 negatives)
- 数据划分: 7:1:2 (train:val:test)
- Seeds: 42, 123, 456, 789, 2024
- Metrics: NDCG@10, Recall@10

## Baseline Results

| Method | NDCG@10 | Recall@10 | KG Type |
|--------|---------|-----------|---------|
| BPR | 0.6655 ± 0.0106 | 0.7208 ± 0.0094 | - |
| LightGCN | 0.6789 ± 0.0048 | 0.7410 ± 0.0026 | - |
| KGAT | 0.6978 ± 0.0184 | 0.8259 ± 0.0036 | metadata (categories + price) |

## Our Method

| Method | NDCG@10 | Recall@10 | KG Type |
|--------|---------|-----------|---------|
| Ours | - | - | LLM-extracted visual KG |

## 数据集统计
- Users: 233,222
- Items: 16,340
- Interactions: 264,801
- Item KG triplets: 88,379
- User KG triplets: 167,035
- Total KG triplets: 255,414

## 更新记录
- 2026-02-01: BPR baseline完成
