# 图谱有效性消融实验

## 目标

证明我们的 **Visual KG（LLM从海报提取）** 比 **Metadata-based KG（从数据集metadata构建）** 更有效。

## 实验设计

对比3个KG推荐方法在2种图谱上的性能：

### 方法
1. **KGAT** - Knowledge Graph Attention Network
2. **KGCN** - Knowledge Graph Convolutional Networks
3. **KGIN** - Knowledge Graph-based Intent Network

### 图谱类型
1. **Metadata-based KG**: 从数据集原始metadata字段构建（genre, director, year等）
   - 数据路径: `/data/xuao/KG4RecEval/dataset/ml-1m`

2. **Visual KG (Ours)**: 用LLM从电影海报中提取的视觉知识图谱
   - 数据路径: `/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m`

### 实验矩阵

| 方法 | Metadata-based KG | Visual KG (Ours) |
|------|-------------------|------------------|
| KGAT | ✅ 已完成 (0.2222) | 📝 待跑 |
| KGCN | 📝 待跑 | 📝 待跑 |
| KGIN | 📝 待跑 | 📝 待跑 |

**总计**: 5个实验需要运行

## 使用方法

### 方式1: 使用Shell脚本（推荐）

```bash
# 在项目根目录下执行
bash scripts/run_kg_ablation.sh
```

脚本会：
1. 进行配置检查（dry run）
2. 列出所有待运行的实验
3. 询问是否继续
4. 依次运行所有5个实验

### 方式2: 直接运行Python脚本

```bash
# 先检查配置（dry run）
python scripts/run_kg_ablation.py --dry-run

# 运行所有实验
python scripts/run_kg_ablation.py
```

## 实验配置

- **数据集**: ML-1M (5-core过滤)
- **评估模式**: uni100 (1 positive + 99 random negatives)
- **Trials**: 1次（seed=42）
- **Epochs**: 300（带early stopping）
- **指标**: NDCG@10, Recall@10
- **GPU**: CUDA

## 输出

### 输出目录结构

```
outputs/kg_ablation/
├── KGAT_visual/           # KGAT + Visual KG
│   ├── run.log
│   └── ml-1m_*/
│       ├── results.json
│       └── checkpoints/
├── KGCN_metadata/         # KGCN + Metadata KG
├── KGCN_visual/           # KGCN + Visual KG
├── KGIN_metadata/         # KGIN + Metadata KG
├── KGIN_visual/           # KGIN + Visual KG
└── kg_ablation_summary.json  # 汇总结果
```

### 结果汇总

实验完成后会生成 `outputs/kg_ablation/kg_ablation_summary.json`，包含所有方法在两种图谱上的结果对比。

示例输出：
```
Method       | KG Type        | NDCG@10    | Recall@10  | Improvement
-------------|----------------|------------|------------|-------------
KGAT         | Metadata       | 0.2222     | 0.1518     | -
KGAT         | Visual (Ours)  | 0.XXXX     | 0.XXXX     | +X.XX%
KGCN         | Metadata       | 0.XXXX     | 0.XXXX     | -
KGCN         | Visual (Ours)  | 0.XXXX     | 0.XXXX     | +X.XX%
...
```

## 预期结果

如果我们的Visual KG更有效，应该看到：
- **Improvement > 0%**: Visual KG在所有方法上都优于Metadata-based KG
- 这将验证LLM提取的视觉知识比传统metadata更有价值

## 时间估算

- 每个实验: ~20-40分钟（300 epochs，带early stopping）
- 总计5个实验: **约2-3小时**

## 故障排除

### 1. 数据路径不存在

**错误**: `Visual KG directory not found` 或 `Metadata KG directory not found`

**解决**: 检查路径是否正确：
- Visual KG: `/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m`
- Metadata KG: `/data/xuao/KG4RecEval/dataset/ml-1m`

### 2. GPU内存不足

**错误**: `CUDA out of memory`

**解决**:
- 减小batch size（修改脚本中的 `train_batch_size`）
- 或者使用CPU: 修改 `--device cuda` 为 `--device cpu`

### 3. RecBole报错

**错误**: `ImportError: No module named 'recbole'`

**解决**:
```bash
pip install recbole
```

## 论文用途

这个消融实验的结果将用于论文中证明：
1. **图谱有效性**: 我们提取的Visual KG本身就比传统Metadata-based KG更好
2. **方法无关性**: 即使用传统KG方法（KGAT, KGCN, KGIN）使用我们的图谱也能获得提升
3. **知识质量**: 证明LLM提取的细粒度视觉知识优于粗粒度的metadata

预期论文章节：
- **Section 4.3**: Ablation Studies - Graph Effectiveness
- **Table X**: Performance comparison of KG methods on different graphs

---

*Created: 2026-01-27*
*Author: Claude Code Assistant*
