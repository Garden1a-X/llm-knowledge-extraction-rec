# 超参数调优指南

本指南介绍如何使用加速训练脚本和并行调优工具来优化模型性能。

## 🚀 快速开始

### 1. 单个加速训练

使用混合精度训练（AMP）和torch.compile加速：

```bash
# 基础用法（使用默认配置）
python scripts/train_model_fast.py --config configs/ours_full.yaml

# 禁用混合精度（如果遇到兼容性问题）
python scripts/train_model_fast.py --config configs/ours_full.yaml --no-amp

# 使用梯度累积（等效更大batch size）
python scripts/train_model_fast.py --config configs/ours_full.yaml --grad-accum-steps 4
# 等效batch size = 2048 * 4 = 8192
```

### 2. 并行超参数调优

同时运行4个不同的超参数配置：

```bash
# 自动模式（根据GPU数量自动选择并行或顺序）
python scripts/run_parallel_tuning.py

# 单GPU模式（顺序执行4个实验）
python scripts/run_parallel_tuning.py --gpu 0

# 多GPU模式（真正并行，每个实验1个GPU）
python scripts/run_parallel_tuning.py --multi-gpu 0,1,2,3

# 非连续GPU（例如0,5,6,7号卡）
python scripts/run_parallel_tuning.py --multi-gpu 0,5,6,7

# 2个GPU（实验会分配到GPU 0和1）
python scripts/run_parallel_tuning.py --multi-gpu 0,1
```

### 3. 对比实验结果

所有实验完成后，对比结果：

```bash
python scripts/compare_tuning_results.py
```

## 📊 4个调优配置说明

| 配置文件 | 聚焦点 | 主要改动 |
|---------|--------|---------|
| `tune1_loss_weights.yaml` | 损失权重组合 | alpha=0.2 ↑, beta=0.1 ↑, gamma=0.005 ↓ |
| `tune2_lr_embed.yaml` | 学习率和嵌入维度 | embed_dim=128 ↑, lr=0.002 ↑ |
| `tune3_depth_dropout.yaml` | 模型深度和正则化 | layers=3 ↑, heads=8 ↑, dropout=0.3 ↑, wd=0.0001 ↑ |
| `tune4_hybrid.yaml` | 混合优化 | 综合以上改进的中间值 |

### 配置对比

```
配置          嵌入维度  GNN层数  注意力头  Dropout  学习率     alpha   beta   gamma
-----------------------------------------------------------------------------
ours_full       64       2        4       0.2     0.001     0.1    0.05   0.01
tune1           64       2        4       0.2     0.001     0.2    0.10   0.005
tune2          128       2        4       0.2     0.002     0.1    0.05   0.01
tune3           64       3        8       0.3     0.001     0.1    0.05   0.01
tune4           96       2        6       0.25    0.0015    0.15   0.08   0.008
```

## ⚡ 加速特性

### 1. 混合精度训练 (AMP)

- 使用`torch.cuda.amp`自动混合精度
- GPU内存减少约40%
- 训练速度提升约2-3倍
- 保持模型精度（float16前向，float32梯度更新）

**适用场景**：
- ✅ 现代GPU（V100, A100, RTX 30/40系列）
- ✅ 需要加速训练的场景
- ❌ 旧GPU或遇到数值不稳定时可以`--no-amp`关闭

### 2. torch.compile

- PyTorch 2.0+新特性
- 自动编译计算图，优化执行
- 首次运行会有编译开销（~1分钟），后续epoch加速明显

**注意**：如果遇到兼容性问题，可以`--no-compile`关闭

### 3. 梯度累积

- 等效更大的batch size，但不增加内存占用
- 例如：batch_size=2048, grad_accum_steps=4 → 等效8192

**适用场景**：
- GPU内存不足时
- 想尝试更大batch size但不想修改配置文件

```bash
python scripts/train_model_fast.py --config configs/ours_full.yaml --grad-accum-steps 4
```

## 📁 输出结构

```
outputs/ours/
└── {model_name}_{dataset}_{date}_{time}/
    ├── checkpoints/
    │   ├── ours_full-Dec-27-2025_13-18-09.pth  (最佳模型)
    │   └── checkpoint_epoch_10.pth              (定期checkpoint)
    └── history.json                              (训练历史)

log/
└── {model_name}/
    └── {run_id}.log                              (训练日志)

log_tensorboard/
└── {run_id}/
    └── events.out.tfevents.*                     (TensorBoard日志)

log/
└── tune_parallel_{config_name}_{timestamp}.log   (并行训练日志)
```

## 📈 监控训练进度

### 1. TensorBoard

```bash
# 查看所有实验
tensorboard --logdir log_tensorboard

# 查看特定实验
tensorboard --logdir log_tensorboard/{run_id}
```

访问: http://localhost:6006

### 2. 实时日志

```bash
# 查看最新日志
tail -f log/ours_full_fast/*.log

# 查看并行调优日志
tail -f log/tune_parallel_*.log
```

### 3. 检查进程状态

```bash
# 查看GPU使用情况
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi

# 查看Python进程
ps aux | grep train_model_fast
```

## 🛠️ 故障排除

### 1. OOM (Out of Memory)

```bash
# 方案1: 使用梯度累积
python scripts/train_model_fast.py --config configs/ours_full.yaml --grad-accum-steps 2

# 方案2: 减小batch size（修改config文件）
# train.batch_size: 2048 → 1024

# 方案3: 减小模型大小（修改config文件）
# model.embedding_dim: 64 → 32
```

### 2. 混合精度训练数值不稳定

```bash
# 禁用AMP
python scripts/train_model_fast.py --config configs/ours_full.yaml --no-amp
```

### 3. torch.compile失败

```bash
# 禁用compile
python scripts/train_model_fast.py --config configs/ours_full.yaml --no-compile
```

### 4. 并行调优中某个实验失败

```bash
# 查看失败实验的日志
cat log/tune_parallel_{config_name}_*.log

# 单独重新运行该实验
python scripts/train_model_fast.py --config configs/{config_name}.yaml
```

## 📊 结果分析

### 对比实验结果

```bash
python scripts/compare_tuning_results.py
```

输出示例：
```
+----------------------+--------------------+-------------+-----------------+-------------------+-----------------+-------------------+
| Config               | Best Val NDCG@10   | Best Epoch  | Test NDCG@10    | Test Recall@10    | Test NDCG@20    | Test Recall@20    |
+======================+====================+=============+=================+===================+=================+===================+
| Tune1: Loss Weights  | 0.1620             | 45          | 0.1580          | 0.2340            | 0.1890          | 0.3120            |
+----------------------+--------------------+-------------+-----------------+-------------------+-----------------+-------------------+
| Tune2: LR & Embed    | 0.1650             | 38          | 0.1610          | 0.2380            | 0.1920          | 0.3150            |
+----------------------+--------------------+-------------+-----------------+-------------------+-----------------+-------------------+
| Tune3: Depth & Drop  | 0.1580             | 52          | 0.1540          | 0.2290            | 0.1850          | 0.3080            |
+----------------------+--------------------+-------------+-----------------+-------------------+-----------------+-------------------+
| Tune4: Hybrid        | 0.1680             | 41          | 0.1640          | 0.2410            | 0.1950          | 0.3190            |
+----------------------+--------------------+-------------+-----------------+-------------------+-----------------+-------------------+

✓ 最佳配置: Tune4: Hybrid
  Test NDCG@10: 0.1640
```

### TensorBoard可视化

打开TensorBoard查看：
- 训练损失曲线
- 各损失分量（L_rec, L_contrast, L_align, L_mask）
- 验证指标曲线（NDCG, Recall等）
- 4个实验的对比

## 🎯 推荐工作流

### 睡前快速调优（用户场景）

```bash
# 1. 启动4个并行实验（指定可用GPU）
# 如果GPU 1,2,3,4被占用，使用0,5,6,7
python scripts/run_parallel_tuning.py --multi-gpu 0,5,6,7

# 或者自动模式（根据可用GPU自动选择）
python scripts/run_parallel_tuning.py

# 2. 检查启动状态
nvidia-smi

# 3. 第二天醒来查看结果
python scripts/compare_tuning_results.py

# 4. TensorBoard详细分析
tensorboard --logdir log_tensorboard
```

### 特殊场景：部分GPU被占用

如果某些GPU正在运行其他任务，可以指定可用的GPU：

```bash
# 示例：只有0,5,6,7号卡可用
python scripts/run_parallel_tuning.py --multi-gpu 0,5,6,7

# GPU分配方案：
#   实验1 (tune1_loss_weights)    → GPU 0
#   实验2 (tune2_lr_embed)        → GPU 5
#   实验3 (tune3_depth_dropout)   → GPU 6
#   实验4 (tune4_hybrid)          → GPU 7

# 如果只有2张卡，也可以运行（会循环分配）
python scripts/run_parallel_tuning.py --multi-gpu 0,5
# 分配：实验1→GPU0, 实验2→GPU5, 实验3→GPU0, 实验4→GPU5
```

### 单次快速实验

```bash
# 使用加速训练（比原版train_model.py快2-3倍）
python scripts/train_model_fast.py --config configs/ours_full.yaml
```

### 消融实验

修改config文件中的`ablation`部分：

```yaml
ablation:
  use_contrast: false  # 禁用对比学习
  use_align: true
  use_mask: true
  use_cf_view: true
  use_kg_view: true
```

## 💡 提示

1. **首次运行**：torch.compile需要编译时间（~1分钟），第2个epoch开始会明显加速
2. **GPU分配**：多实验并行时，确保每个实验有足够显存（建议每个实验4-6GB）
3. **Early Stopping**：patience=10，如果10个epoch内验证指标没提升会自动停止
4. **随机种子**：4个配置使用不同随机种子（42,43,44,45）确保结果多样性
5. **日志保存**：所有日志都会保存，可以随时查看历史实验

## 🔗 相关文件

- 加速训练脚本: `scripts/train_model_fast.py`
- 并行调优脚本: `scripts/run_parallel_tuning.py`
- 结果对比脚本: `scripts/compare_tuning_results.py`
- 原始训练脚本: `scripts/train_model.py`（未加速版本）
- 配置文件目录: `configs/`
