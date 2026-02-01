# Amazon Beauty 实验结果汇总

## 实验设置
- 数据集: Amazon Beauty (无5-core过滤)
- 评估模式: uni100 (1 positive + 99 negatives)
- 数据划分: 7:1:2 (train:val:test), per-user temporal split
- Seeds: 42, 123, 456, 789, 2024
- Metrics: NDCG@10, Recall@10

## 重要发现: 数据划分不一致问题

RecBole使用的数据划分与我们的`split_data()`函数不同，导致结果不可比。

验证方法: 使用相同的BPR实现，分别用RecBole划分和我们的划分进行对比:
- RecBole划分 BPR: NDCG@10=0.6655, Recall@10=0.7208
- 我们的划分 BPR: NDCG@10=0.5120, Recall@10=0.5941

**结论**: 必须使用我们的划分重新运行所有baseline。

## Baseline Results (使用我们的数据划分)

### CF Baselines
| Method | NDCG@10 | Recall@10 | 备注 |
|--------|---------|-----------|------|
| BPR | 0.5089 ± 0.0041 | 0.5928 ± 0.0072 | - |
| LightGCN | 0.5311 ± 0.0098 | 0.6106 ± 0.0154 | - |
| KGAT | 0.5185 ± 0.0087 | 0.5998 ± 0.0149 | metadata KG |

### Multimodal Baselines
| Method | NDCG@10 | Recall@10 | 备注 |
|--------|---------|-----------|------|
| VBPR | - | - | visual features |
| MMGCN | - | - | visual features + GCN |

## Baseline Results (RecBole默认划分 - 仅供参考，不可比)

| Method | NDCG@10 | Recall@10 | KG Type |
|--------|---------|-----------|---------|
| BPR | 0.6655 ± 0.0106 | 0.7208 ± 0.0094 | - |
| LightGCN | 0.6789 ± 0.0048 | 0.7410 ± 0.0026 | - |
| KGAT | 0.6978 ± 0.0184 | 0.8259 ± 0.0036 | metadata |

## Our Method

| Method | NDCG@10 | Recall@10 | KG Type |
|--------|---------|-----------|---------|
| Ours | 0.5789 ± 0.0127 | 0.7219 ± 0.0121 | LLM-extracted visual + user KG |

## 数据集统计
- Users: 233,222
- Items: 16,340
- Interactions: 264,801
- Item KG triplets: 88,379
- User KG triplets: 167,035
- Total KG triplets: 255,414

## 训练脚本

使用我们数据划分的baseline脚本:
- `scripts/train_bpr_our_split.py` - BPR
- `scripts/train_lightgcn_our_split.py` - LightGCN
- `scripts/train_kgat_our_split.py` - KGAT (使用metadata KG)
- `scripts/train_vbpr_our_split.py` - VBPR (使用visual features)
- `scripts/train_mmgcn_our_split.py` - MMGCN (使用visual features)

运行方式:
```bash
# 单个方法
python scripts/train_bpr_our_split.py --seed 42
python scripts/train_lightgcn_our_split.py --seed 42
python scripts/train_kgat_our_split.py --seed 42
python scripts/train_vbpr_our_split.py --seed 42
python scripts/train_mmgcn_our_split.py --seed 42

# 5-trial批量运行
python scripts/run_baselines_our_split_5trials.py --method all
python scripts/run_baselines_our_split_5trials.py --method multimodal  # 只跑VBPR和MMGCN
```

## 更新记录
- 2026-02-01: 发现数据划分不一致问题
- 2026-02-01: 创建使用我们划分的baseline训练脚本
- 2026-02-01: Ours 5-trial完成: NDCG@10=0.5789±0.0127, Recall@10=0.7219±0.0121
- 2026-02-01: Baseline 5-trial完成 (BPR, LightGCN, KGAT)
