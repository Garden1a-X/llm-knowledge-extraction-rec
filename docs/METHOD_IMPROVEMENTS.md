# 方法改进方向 - 计划文档（未实施）

**状态**: ⏸️ 计划中，未实施
**创建时间**: 2026-01-12晚
**说明**: 此文档记录的改进方向（负采样、动态温度、双层对比）尚未实施，作为未来可选的改进方向保留

---

## 🎯 背景

当前baseline：**Ours-Full NDCG@10 = 0.1549**
- 超越BPR +27.1%
- 超越LightGCN +22.3%

但第一轮调参失败（tune2-4都没提升），说明需要**方法论层面的改进**，而不是简单的超参数调整。

---

## 💡 改进方向（按优先级）

### **方向1: 负采样策略改进** ⭐⭐⭐

#### 当前实现
```python
# src/data/dataset.py
num_negatives = 1  # 每个正样本只采1个负样本
# 随机采样，没有hard negative mining
```

#### 问题
- 负样本太少（1个），模型学不到足够的区分性
- 随机采样可能采到太简单的负样本（cold items）
- RecBole的BPR也是1个负样本，但我们的模型更复杂

#### 改进方案A: **增加负样本数量**

**代码修改**：
```python
# configs/ours_full_neg4.yaml
train:
  num_negatives: 4  # 从1增加到4
  batch_size: 1024  # 从2048减到1024（因为负样本多了）
```

**预期效果**：
- 更多负样本 → 更强的对比学习
- BPR loss更稳定
- 可能提升2-5%

**实现难度**：⭐（改配置即可）
**训练时间**：略增（~1.5倍）

---

#### 改进方案B: **Hard Negative Sampling**

**思路**：采样与正样本相似度高的负样本（harder to distinguish）

**实现步骤**：

1. **预计算item相似度矩阵**（基于用户共现）
```python
# scripts/compute_item_similarity.py
def compute_item_similarity(inter_df):
    """计算item-item相似度（基于共同用户数）"""
    # 使用Jaccard相似度或cosine相似度
    # 保存到: data/item_similarity.npy
```

2. **修改负采样策略**
```python
# src/data/dataset.py
def _hard_negative_sampling(self, user_id, pos_item, num_neg):
    """
    采样策略：
    - 50% hard negatives（与pos_item相似的items）
    - 50% random negatives（保持多样性）
    """
    pos_items = self.user_items[user_id]

    # Hard negatives: 与pos_item相似但用户未交互
    similar_items = self.item_similarity[pos_item]  # top-K similar
    hard_candidates = [i for i in similar_items if i not in pos_items]

    # 50-50混合
    n_hard = num_neg // 2
    n_random = num_neg - n_hard

    hard_negs = sample(hard_candidates, n_hard)
    random_negs = sample(all_items - pos_items, n_random)

    return hard_negs + random_negs
```

**预期效果**：
- 更难的负样本 → 模型学到更细粒度的区分
- 可能提升3-8%

**实现难度**：⭐⭐（需要新增代码）
**训练时间**：不变（采样更智能但数量不变）

---

### **方向2: Attention机制改进** ⭐⭐

#### 当前实现
```python
# src/model/encoders.py
# 使用标准的GAT（GATConv）
# 所有边的attention都是独立计算的
```

#### 问题
- User-Entity边（long-term vs short-term）没有区分
- 可能short-term interest应该有更高的attention weight

#### 改进方案: **Temporal-Aware Attention**

**思路**：给short-term interest更高的权重

```python
# src/model/encoders.py - KGEncoder
class TemporalKGEncoder(nn.Module):
    def __init__(self, ...):
        # 为long-term和short-term学习不同的权重
        self.edge_type_weight = nn.Parameter(torch.ones(2))  # [long, short]

    def forward(self, x_dict, edge_index_dict):
        # 分别处理long-term和short-term边
        long_term_edges = edge_index_dict[('user', 'long_term', 'entity')]
        short_term_edges = edge_index_dict[('user', 'short_term', 'entity')]

        # 加权聚合
        h_long = self.conv_long(x, long_term_edges) * self.edge_type_weight[0]
        h_short = self.conv_short(x, short_term_edges) * self.edge_type_weight[1]

        return h_long + h_short
```

**预期效果**：
- 模型自动学习long-term vs short-term的平衡
- 可能提升1-3%

**实现难度**：⭐⭐⭐（需要修改编码器）
**训练时间**：不变

---

### **方向3: 对比学习改进** ⭐⭐

#### 当前实现
```python
# src/model/losses.py
# 使用简单的InfoNCE loss
# CF view vs KG view对比
```

#### 问题
- 只在user embedding层面做对比
- item embedding没有对比学习

#### 改进方案: **双层对比学习**

**思路**：同时在user和item层面做对比

```python
# src/model/losses.py
class ImprovedContrastLoss(nn.Module):
    def forward(self, outputs):
        # 原有：User-level对比
        L_contrast_user = self.infonce(
            outputs['user_cf'],
            outputs['user_kg']
        )

        # 新增：Item-level对比
        L_contrast_item = self.infonce(
            outputs['item_cf'],
            outputs['item_kg']
        )

        return L_contrast_user + 0.5 * L_contrast_item
```

**预期效果**：
- Item embedding也能享受多视图对比的好处
- 可能提升2-4%

**实现难度**：⭐⭐（修改损失函数）
**训练时间**：不变

---

### **方向4: 温度参数动态调整** ⭐⭐

#### 当前实现
```python
# configs/ours_full.yaml
loss:
  temperature_rec: 0.2  # 固定
  temperature_contrast: 0.1  # 固定
```

#### 问题
- 固定温度在训练初期和后期都一样
- 训练初期应该高温（exploration），后期低温（exploitation）

#### 改进方案: **Cosine Annealing Temperature**

```python
# src/model/losses.py
class DynamicTemperatureLoss(nn.Module):
    def __init__(self, temp_init=0.5, temp_final=0.1):
        self.temp_init = temp_init
        self.temp_final = temp_final

    def get_temperature(self, epoch, max_epochs):
        # Cosine annealing
        temp = self.temp_final + 0.5 * (self.temp_init - self.temp_final) * \
               (1 + np.cos(np.pi * epoch / max_epochs))
        return temp
```

**预期效果**：
- 训练初期：高温，模型探索更多
- 训练后期：低温，模型收敛更好
- 可能提升1-3%

**实现难度**：⭐⭐（修改损失函数）
**训练时间**：不变

---

## 🔧 实施优先级排序

### **Tier 1: 快速见效（明天优先实现）**

1. **增加负样本数量** ⭐⭐⭐
   - 实现难度：⭐
   - 预期收益：2-5%
   - 时间成本：30分钟

2. **温度参数动态调整** ⭐⭐
   - 实现难度：⭐⭐
   - 预期收益：1-3%
   - 时间成本：1-2小时

3. **双层对比学习** ⭐⭐
   - 实现难度：⭐⭐
   - 预期收益：2-4%
   - 时间成本：2小时

### **Tier 2: 值得尝试（如果时间允许）**

4. **Hard Negative Sampling** ⭐⭐
   - 实现难度：⭐⭐
   - 预期收益：3-8%
   - 时间成本：3-4小时

5. **Temporal-Aware Attention** ⭐⭐
   - 实现难度：⭐⭐⭐
   - 预期收益：1-3%
   - 时间成本：3-4小时

---

## 📋 明天的实施计划

### **上午（9:00-12:00）: 快速改进**

#### 改进1: 增加负样本数量（30分钟）
```bash
# 1. 创建新配置
cp configs/ours_full.yaml configs/ours_neg4.yaml

# 2. 修改配置
# train.num_negatives: 1 → 4
# train.batch_size: 2048 → 1024

# 3. 运行训练
CUDA_VISIBLE_DEVICES=0 python scripts/train_model_fast.py \
    --config configs/ours_neg4.yaml &
```

#### 改进2: 温度动态调整（1-2小时）
1. 修改`src/model/losses.py`
2. 添加`get_temperature()`方法
3. 在`train_model_fast.py`中传入当前epoch
4. 创建配置并运行

#### 改进3: 双层对比学习（2小时）
1. 修改`src/model/losses.py`的`ContrastLoss`
2. 添加item-level对比
3. 创建配置并运行

**预计完成时间**: 12:00

---

### **下午（14:00-17:00）: 结果分析 + 可选改进**

#### 任务1: 分析上午3个改进的结果
- 对比baseline (0.1549)
- 找出最有效的改进
- 决定是否组合多个改进

#### 任务2: 如果时间允许，实现Hard Negative Sampling
1. 计算item相似度矩阵（基于共现）
2. 修改dataset.py的负采样逻辑
3. 运行训练

---

### **晚上（18:00-20:00）: 消融实验**

无论上午改进是否成功，都要运行消融实验（这是论文核心）：
```bash
./run_ablation_experiments.sh
```

---

## 🎯 预期最终结果

如果3个Tier 1改进都有效（保守估计）：
- 当前：0.1549
- 改进1（负样本×4）：+3% → 0.1595
- 改进2（动态温度）：+2% → 0.1627
- 改进3（双层对比）：+2% → 0.1660

**最终预期：NDCG@10 = 0.1650-0.1700**
- vs BPR: +35-40%
- vs LightGCN: +30-34%

这将是非常显著的提升！

---

## 💭 为什么这些改进比调参更有价值？

### 调参的问题：
- ❌ tune3（深度+dropout）：过拟合，性能下降20%
- ❌ tune4（综合策略）：收敛过快，性能下降10%
- ⚠️ tune2（LR+embed）：持平，没有提升

**结论**：简单的超参数调整已经到瓶颈

### 方法改进的优势：
- ✅ **负采样改进**：解决训练信号不足的问题
- ✅ **温度调整**：改善训练动态过程
- ✅ **双层对比**：充分利用多视图信息
- ✅ **Hard Negative**：提升模型判别能力

这些是**架构和训练策略层面的改进**，比调超参数更fundamental！

---

## 🔗 相关论文支持

1. **负采样**：
   - "Revisiting Negative Sampling for BPR" (RecSys 2020)
   - 结论：4-8个负样本最优

2. **Hard Negative**：
   - "Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations" (WWW 2023)
   - 结论：混合hard+random效果最好

3. **温度调整**：
   - "Understanding the Behaviour of Contrastive Loss" (CVPR 2021)
   - 结论：动态温度优于固定温度

4. **双层对比**：
   - "Contrastive Learning for Representation Degeneration Problem" (WSDM 2022)
   - 结论：多层对比学习缓解退化

---

## 📝 实现Checklist

### 改进1: 负样本数量
- [ ] 创建configs/ours_neg4.yaml
- [ ] 修改num_negatives: 1 → 4
- [ ] 修改batch_size: 2048 → 1024
- [ ] 运行训练
- [ ] 记录结果

### 改进2: 动态温度
- [ ] 修改src/model/losses.py
- [ ] 添加get_temperature()方法
- [ ] 在RecommendationLoss中集成
- [ ] 修改train_model_fast.py传入epoch参数
- [ ] 运行训练
- [ ] 记录结果

### 改进3: 双层对比
- [ ] 修改src/model/losses.py
- [ ] 添加item-level对比损失
- [ ] 创建配置
- [ ] 运行训练
- [ ] 记录结果

### 改进4: Hard Negative（可选）
- [ ] 实现scripts/compute_item_similarity.py
- [ ] 修改src/data/dataset.py
- [ ] 添加_hard_negative_sampling()方法
- [ ] 运行训练
- [ ] 记录结果

---

**创建时间**: 2026-01-13 凌晨
**优先级**: 方法改进 > 消融实验 > 多种子验证
**预期收益**: +5-10% NDCG@10提升
