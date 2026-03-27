# 模型训练指南

*Last updated: 2026-01-12*

---

## 📋 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision torch-geometric
pip install pandas numpy pyyaml tensorboard tqdm
```

### 2. 运行训练

```bash
# 完整模型
python scripts/train_model.py --config configs/ours_full.yaml

# 消融实验
python scripts/train_model.py --config configs/ours_wo_contrast.yaml
python scripts/train_model.py --config configs/ours_wo_mask.yaml
python scripts/train_model.py --config configs/ours_kg_only.yaml
```

### 3. 查看结果

```bash
# TensorBoard
tensorboard --logdir outputs/ours_full/tensorboard

# 训练历史
cat outputs/ours_full/history.json
```

---

## 📁 项目结构

```
llm-knowledge-extraction-rec/
├── src/
│   ├── data/
│   │   ├── graph_builder.py      # 图构建
│   │   └── dataset.py             # 数据集
│   ├── model/
│   │   ├── encoders.py            # CF/KG编码器
│   │   ├── losses.py              # 损失函数
│   │   └── ours.py                # 完整模型
│   └── utils/
│       ├── metrics.py             # 评估指标
│       └── config.py              # 配置管理
├── scripts/
│   └── train_model.py             # 训练脚本
├── configs/
│   ├── ours_full.yaml             # 完整模型配置
│   ├── ours_wo_contrast.yaml      # 无对比学习
│   ├── ours_wo_mask.yaml          # 无Mask
│   └── ours_kg_only.yaml          # 只用KG视图
├── data/recbole/ml-1m/
│   ├── ml-1m.item.kg              # Item知识图谱
│   ├── ml-1m.user.kg              # User兴趣图谱
│   └── ml-1m.inter                # 交互数据
└── outputs/                       # 训练输出
    └── {experiment_name}/
        ├── tensorboard/           # TensorBoard日志
        ├── history.json           # 训练历史
        └── checkpoints/           # 模型checkpoint
```

---

## ⚙️ 配置文件说明

### 基本配置结构

```yaml
name: ours_full                    # 实验名称
description: Full model            # 描述

data:                              # 数据配置
  item_kg_path: ...                # Item KG路径
  user_kg_path: ...                # User KG路径
  inter_path: ...                  # 交互数据路径
  train_ratio: 0.7                 # 训练集比例
  val_ratio: 0.1                   # 验证集比例
  test_ratio: 0.2                  # 测试集比例
  time_based_split: true           # 基于时间分割

model:                             # 模型配置
  embedding_dim: 64                # Embedding维度
  num_gnn_layers: 2                # GNN层数
  gat_heads: 4                     # GAT attention heads
  dropout: 0.2                     # Dropout
  use_mask: true                   # 是否使用Mask

loss:                              # 损失函数配置
  alpha_contrast: 0.1              # 多视图对比权重
  beta_align: 0.05                 # Entity-Item对齐权重
  gamma_mask: 0.01                 # Mask正则权重
  temperature_rec: 0.2             # InfoNCE温度
  temperature_contrast: 0.1        # 对比学习温度

train:                             # 训练配置
  batch_size: 1024                 # Batch大小
  learning_rate: 0.001             # 学习率
  num_epochs: 300                  # 训练轮数
  early_stop_patience: 20          # Early stopping
  eval_every: 5                    # 评估频率
  device: cuda                     # 设备

ablation:                          # 消融实验开关
  use_contrast: true               # 是否使用多视图对比
  use_align: true                  # 是否使用对齐损失
  use_mask: true                   # 是否使用Mask
  use_cf_view: true                # 是否使用CF视图
  use_kg_view: true                # 是否使用KG视图
```

---

## 🧪 消融实验

### 1. 损失函数组件

```bash
# 完整模型
python scripts/train_model.py --config configs/ours_full.yaml

# 去掉多视图对比
python scripts/train_model.py --config configs/ours_wo_contrast.yaml

# 去掉Mask机制
python scripts/train_model.py --config configs/ours_wo_mask.yaml
```

### 2. 视图选择

```bash
# 只用KG视图
python scripts/train_model.py --config configs/ours_kg_only.yaml

# 只用CF视图（需要创建配置）
python scripts/train_model.py --config configs/ours_cf_only.yaml
```

### 3. 并行实验（4张GPU）

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python scripts/train_model.py --config configs/ours_full.yaml

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python scripts/train_model.py --config configs/ours_wo_contrast.yaml

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python scripts/train_model.py --config configs/ours_wo_mask.yaml

# Terminal 4
CUDA_VISIBLE_DEVICES=3 python scripts/train_model.py --config configs/ours_kg_only.yaml
```

---

## 📊 输出说明

### 训练日志

```
Epoch 50/300
  Train Loss: 0.3254
    L_rec: 0.2891
    L_contrast: 0.0245
    L_align: 0.0098
    L_mask: 0.0020
  Val Metrics:
    NDCG@10: 0.1234
    Recall@10: 0.2345
  ✓ New best model! NDCG@10=0.1234
```

### history.json

```json
{
  "train_loss": [0.45, 0.42, 0.38, ...],
  "train_L_rec": [0.40, 0.37, 0.33, ...],
  "val_NDCG@10": [0.10, 0.11, 0.12, ...],
  "val_Recall@10": [0.20, 0.21, 0.23, ...],
  "final_test_metrics": {
    "NDCG@10": 0.1234,
    "Recall@10": 0.2345,
    ...
  },
  "best_epoch": 95,
  "best_ndcg": 0.1234
}
```

### Checkpoint

```python
checkpoint = {
    'epoch': 95,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'best_ndcg': 0.1234,
    'metrics': {...},
    'config': {...}
}
```

---

## 🔧 常见问题

### Q: CUDA out of memory

**解决**：减小batch size
```yaml
train:
  batch_size: 512  # 从1024减到512
```

### Q: 训练太慢

**解决**：
1. 减少workers数量（如果I/O瓶颈）
2. 减少GNN层数
3. 减少GAT heads

### Q: 验证指标不上升

**解决**：
1. 检查学习率（可能太大或太小）
2. 检查数据分割是否正确
3. 尝试调整损失权重（alpha, beta, gamma）

---

## 📈 性能预期

### ML-1M数据集

| 配置 | NDCG@10 | Recall@10 | 训练时间（300 epochs）| 状态 |
|------|---------|-----------|----------------------|------|
| Ours-Full | **0.1549** | **0.0722** | ~6分钟（单卡，AMP） | ✅ 完成 |
| Ours w/o Contrast | 待运行 | 待运行 | ~5分钟 | ⏸️ 配置就绪 |
| Ours w/o Mask | 待运行 | 待运行 | ~6分钟 | ⏸️ 配置就绪 |
| Ours KG-only | 待运行 | 待运行 | ~5分钟 | ⏸️ 配置就绪 |
| Ours CF-only | 待运行 | 待运行 | ~5分钟 | ⏸️ 配置就绪 |

**Baseline对比**：
- BPR: 0.1219 (RecBole)
- LightGCN: 0.1267 (RecBole)
- KGAT: 0.1209 (RecBole)
- **Ours-Full: 0.1549** (+27.1% vs BPR) ⭐

*注：使用混合精度训练(AMP)加速2-3x*

---

## 🎯 下一步

1. **消融实验** ✅ 配置就绪
   - 运行ours_wo_contrast, ours_wo_mask, ours_kg_only, ours_cf_only
   - 对比分析各组件贡献

2. **可视化分析**：
   - Mask权重分布
   - 训练曲线对比
   - Embedding t-SNE可视化
   - Attention权重可视化

3. **结果整理**：
   - 创建对比表格
   - 撰写分析报告
   - 论文实验部分撰写

4. **方法改进**（可选）：
   - 负采样策略改进
   - 动态温度调整
   - 双层对比学习

---

*Good luck with your experiments! 🚀*
