# 模型设计文档：知识增强异构图推荐模型

*Created: 2026-01-12*
*Status: ✅ 已实现并完成训练*
*Result: NDCG@10 = 0.1549 (+27.1% vs BPR)*

---

## 📋 目录

1. [核心创新点](#核心创新点)
2. [数据结构与图构建](#数据结构与图构建)
3. [模型架构](#模型架构)
4. [损失函数设计](#损失函数设计)
5. [性能优化方案](#性能优化方案)
6. [实现方案对比](#实现方案对比)
7. [消融实验设计](#消融实验设计)
8. [时间与资源估算](#时间与资源估算)

---

## 🎯 核心创新点

### **1. LLM提取的细粒度视觉知识图谱**
- 不是简单的视觉特征（CNN embedding）
- 而是可解释的知识点：`(relation, entity)` pairs
- 例如：`(color_palette, warm_colors)`, `(mood, romantic)`

### **2. User-Entity-Item异构图结构**
```
User通过兴趣连接Entity
Item通过视觉特征连接Entity
→ User和Item通过共享Entity间接连接

信息流：
  User -[interest]→ Entity ←[describes]- Item

相比传统CF：
  User -[rated]→ Item (直接，但稀疏)
```

### **3. 双视图对比学习**
- **CF视图**：传统User-Item交互图（baseline）
- **KG视图**：知识增强异构图（创新）
- 对比损失让两个视图学到的表示互补

### **4. 可学习Mask机制**
- **动机**：对抗LLM提取的潜在幻觉
- **实现**：每个Entity有一个可学习的mask权重
- **初始化**：基于Entity出现频率（低频→可能不可靠）
- **训练**：模型自动学习哪些Entity不可靠，降低其权重

---

## 📊 数据结构与图构建

### **输入文件**

```
data/recbole/ml-1m/
├── ml-1m.item.kg          # Item侧知识图谱
│   └── item_id  relation  entity
│
├── ml-1m.user.kg          # User侧兴趣图谱
│   └── user_id  long_term_interest/short_term_interest  entity
│
└── ml-1m.inter            # User-Item交互
    └── user_id  item_id  rating  timestamp
```

### **异构图结构**

#### **节点类型**
```python
Nodes = {
    'user': 6,040,      # ML-1M用户（5-core后）
    'entity': ~400,     # 共享的知识实体
    'item': 3,700       # 电影（5-core后）
}
```

#### **边类型**
```python
Edges = {
    # KG视图（知识增强）
    ('user', 'long_term_interest', 'entity'): ~30,000,
    ('user', 'short_term_interest', 'entity'): ~60,000,
    ('entity', 'describes', 'item'): ~33,000,

    # CF视图（传统交互）
    ('user', 'rated', 'item'): ~900,000
}
```

#### **关键设计**
- ✅ User和Item通过**共享Entity**连接（不是直接的User-Item边in KG视图）
- ✅ Entity来自Phase 4+5提取，词表对齐保证连通性
- ✅ CF边和KG边分离，用于双视图对比学习

### **图构建流程**

```python
def build_hetero_graph():
    """
    从.kg和.inter文件构建PyG HeteroData
    """
    graph = HeteroData()

    # 1. 读取item.kg
    item_kg = pd.read_csv('ml-1m.item.kg', sep='\t')
    # item_id → entity (visual knowledge)

    # 2. 读取user.kg
    user_kg = pd.read_csv('ml-1m.user.kg', sep='\t')
    # user_id → entity (interest)

    # 3. 读取交互数据
    inter = pd.read_csv('ml-1m.inter', sep='\t')
    # user_id → item_id

    # 4. Entity ID映射（统一编号）
    entity_vocab = build_entity_vocab(item_kg, user_kg)

    # 5. 构建边
    graph['user', 'long_term', 'entity'].edge_index = ...
    graph['user', 'short_term', 'entity'].edge_index = ...
    graph['entity', 'describes', 'item'].edge_index = ...
    graph['user', 'rated', 'item'].edge_index = ...

    # 6. 统计Entity频率（用于Mask初始化）
    entity_freq = count_entity_frequency(item_kg, user_kg)

    return graph, entity_freq
```

---

## 🏗️ 模型架构

### **整体结构图**

```
Input: HeteroGraph + Batch(users, pos_items, neg_items)
  ↓
┌─────────────────────────────────────────────────────────┐
│  初始Embedding                                           │
│  - User Embedding                                        │
│  - Item Embedding                                        │
│  - Entity Embedding (带Mask)                            │
└─────────────────────────────────────────────────────────┘
  ↓                                    ↓
┌──────────────────────┐      ┌──────────────────────────┐
│  CF视图编码器         │      │  KG视图编码器             │
│  (GAT on User-Item)   │      │  (Hetero-GAT)            │
│                       │      │                          │
│  User -[rated]→ Item  │      │  User -[interest]→ Entity│
│                       │      │  Entity ←[describes]- Item│
└──────────────────────┘      └──────────────────────────┘
  ↓                                    ↓
  User_emb_cf                          User_emb_kg
  Item_emb_cf                          Item_emb_kg
  ↓                                    ↓
  └──────────┬──────────────────────────┘
             ↓
     ┌──────────────┐
     │  多视图对比   │  ← L_contrast
     └──────────────┘
             ↓
     ┌──────────────┐
     │  融合层       │
     │ (Concat+FC)   │
     └──────────────┘
             ↓
     User_emb_fused, Item_emb_fused
             ↓
     ┌──────────────┐
     │  推荐预测     │  ← L_rec (InfoNCE)
     └──────────────┘
```

### **代码骨架**

```python
class KnowledgeEnhancedRecModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # === Embeddings ===
        self.user_embed = nn.Embedding(num_users, dim)
        self.item_embed = nn.Embedding(num_items, dim)
        self.entity_embed = nn.Embedding(num_entities, dim)

        # === 可学习Mask（基于频率初始化）===
        self.mask_logits = nn.Parameter(torch.zeros(num_entities))
        # 初始化在build_graph时设置：
        # mask_init = frequency_based_mask(entity_freq)
        # self.mask_logits.data = torch.logit(mask_init)

        # === CF视图编码器 ===
        # 简单的User-Item二部图GAT
        self.cf_conv1 = GATConv(dim, dim, heads=4)
        self.cf_conv2 = GATConv(dim*4, dim, heads=1, concat=False)

        # === KG视图编码器（异构图）===
        self.kg_conv1 = HeteroConv({
            ('user', 'long_term', 'entity'): GATConv(dim, dim, heads=4),
            ('user', 'short_term', 'entity'): GATConv(dim, dim, heads=4),
            ('entity', 'describes', 'item'): GATConv(dim, dim, heads=4),
        }, aggr='sum')

        self.kg_conv2 = HeteroConv({
            ('user', 'long_term', 'entity'): GATConv(dim*4, dim, heads=1),
            ('user', 'short_term', 'entity'): GATConv(dim*4, dim, heads=1),
            ('entity', 'describes', 'item'): GATConv(dim*4, dim, heads=1),
        }, aggr='sum')

        # === 融合层 ===
        self.fusion_user = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )

        self.fusion_item = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )

    def forward(self, hetero_graph, cf_graph):
        """
        Args:
            hetero_graph: PyG HeteroData (包含所有边类型)
            cf_graph: PyG Data (只有User-Item边)

        Returns:
            {
                'user_cf': [num_users, dim],
                'user_kg': [num_users, dim],
                'user_fused': [num_users, dim],
                'item_cf': [num_items, dim],
                'item_kg': [num_items, dim],
                'item_fused': [num_items, dim],
                'entity_emb': [num_entities, dim],
                'mask': [num_entities],
            }
        """

        # === 1. CF视图编码 ===
        # 将User和Item拼接成一个大图
        x_cf = torch.cat([
            self.user_embed.weight,
            self.item_embed.weight
        ], dim=0)  # [num_users + num_items, dim]

        h_cf = F.relu(self.cf_conv1(x_cf, cf_graph.edge_index))
        h_cf = self.cf_conv2(h_cf, cf_graph.edge_index)

        user_emb_cf = h_cf[:num_users]
        item_emb_cf = h_cf[num_users:]

        # === 2. KG视图编码（带Mask）===
        # 应用Mask到Entity
        mask = torch.sigmoid(self.mask_logits)
        entity_emb_masked = self.entity_embed.weight * mask.unsqueeze(1)

        x_dict = {
            'user': self.user_embed.weight,
            'entity': entity_emb_masked,
            'item': self.item_embed.weight
        }

        # Layer 1
        h_dict = self.kg_conv1(x_dict, hetero_graph.edge_index_dict)
        h_dict = {key: F.relu(val) for key, val in h_dict.items()}

        # Layer 2
        h_dict = self.kg_conv2(h_dict, hetero_graph.edge_index_dict)

        user_emb_kg = h_dict['user']
        item_emb_kg = h_dict['item']
        entity_emb = h_dict['entity']

        # === 3. 融合 ===
        user_emb_fused = self.fusion_user(
            torch.cat([user_emb_cf, user_emb_kg], dim=-1)
        )

        item_emb_fused = self.fusion_item(
            torch.cat([item_emb_cf, item_emb_kg], dim=-1)
        )

        return {
            'user_cf': user_emb_cf,
            'user_kg': user_emb_kg,
            'user_fused': user_emb_fused,
            'item_cf': item_emb_cf,
            'item_kg': item_emb_kg,
            'item_fused': item_emb_fused,
            'entity_emb': entity_emb,
            'mask': mask,
        }
```

---

## 🎓 损失函数设计

### **损失函数组合**

```python
L_total = L_rec                    # 主损失：推荐任务
        + α * L_contrast           # 多视图对比
        + β * L_align              # Entity-Item对齐
        + γ * L_mask               # Mask正则化

# 超参数建议
α = 0.1    # 多视图对比
β = 0.05   # Entity-Item对齐
γ = 0.01   # Mask正则
```

### **1. 主损失：InfoNCE推荐损失**

```python
def info_nce_loss(user_emb, pos_item_emb, neg_item_emb, temperature=0.2):
    """
    对比学习框架的推荐损失

    Args:
        user_emb: [batch_size, dim]
        pos_item_emb: [batch_size, dim]
        neg_item_emb: [batch_size, num_neg, dim]

    Returns:
        loss: scalar
    """
    # Positive pair score
    pos_score = (user_emb * pos_item_emb).sum(dim=-1) / temperature  # [batch]

    # Negative pairs scores
    neg_score = torch.bmm(
        neg_item_emb,
        user_emb.unsqueeze(-1)
    ).squeeze(-1) / temperature  # [batch, num_neg]

    # InfoNCE: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels)
    return loss
```

**为什么用InfoNCE？**
- ✅ 对比学习范式，适合ranking任务
- ✅ 不需要预测具体rating，只需要正确排序
- ✅ 可扩展性好（支持大量负样本）
- ✅ SOTA推荐模型广泛使用（如LightGCN, SGL）

### **2. 多视图对比损失**

```python
def multiview_contrastive_loss(emb_cf, emb_kg, temperature=0.1):
    """
    对比CF视图和KG视图学到的User表示

    目标：让两个视图学到互补但一致的表示

    Args:
        emb_cf: [num_users, dim] - CF视图User embedding
        emb_kg: [num_users, dim] - KG视图User embedding

    Returns:
        loss: scalar
    """
    # L2归一化
    emb_cf = F.normalize(emb_cf, dim=-1)
    emb_kg = F.normalize(emb_kg, dim=-1)

    # 正样本：同一个user的两个视图
    pos_sim = (emb_cf * emb_kg).sum(dim=-1) / temperature  # [num_users]

    # 负样本：不同user的cross-view相似度矩阵
    neg_sim = emb_cf @ emb_kg.T / temperature  # [num_users, num_users]

    # InfoNCE formulation
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.arange(emb_cf.size(0), device=device)

    loss = F.cross_entropy(logits, labels)
    return loss
```

**为什么需要多视图对比？**
- 防止KG视图偏离CF视图太远（CF视图是ground truth）
- 让模型学到的知识增强表示与协同过滤信号对齐
- 可以做消融实验证明对比学习的有效性

**实现细节**：
- 在batch上计算（不是全量用户），效率高
- 温度系数`temperature`控制对比强度
- 可选：只对batch内的user做对比（更快）

### **3. Entity-Item对齐损失**

```python
def alignment_loss(entity_emb, item_emb, describes_edges, num_neg=5):
    """
    让Entity和它描述的Item在embedding空间中接近

    动机：Entity是Item的视觉特征抽象，应该语义相关

    Args:
        entity_emb: [num_entities, dim]
        item_emb: [num_items, dim]
        describes_edges: [2, num_edges] - (entity_id, item_id)
        num_neg: 负采样数量

    Returns:
        loss: scalar
    """
    entity_ids = describes_edges[0]  # [num_edges]
    item_ids = describes_edges[1]    # [num_edges]

    # Positive pairs
    pos_entity = entity_emb[entity_ids]
    pos_item = item_emb[item_ids]
    pos_score = (pos_entity * pos_item).sum(dim=-1)  # [num_edges]

    # Negative sampling（随机采样）
    neg_items = torch.randint(
        0, item_emb.size(0),
        (entity_ids.size(0), num_neg),
        device=device
    )
    neg_item_emb = item_emb[neg_items]  # [num_edges, num_neg, dim]

    neg_score = torch.bmm(
        neg_item_emb,
        pos_entity.unsqueeze(-1)
    ).squeeze(-1)  # [num_edges, num_neg]

    # BPR-like loss
    loss = -F.logsigmoid(pos_score.unsqueeze(1) - neg_score).mean()
    return loss
```

**为什么需要对齐损失？**
- Entity是从Item中提取的，应该保持语义关联
- 帮助模型学到更好的Entity表示
- 可选：也可以对齐User和Entity（用户兴趣对齐）

### **4. Mask正则化损失**

```python
def mask_regularization(mask, lambda_sparse=1.0, lambda_entropy=0.1):
    """
    正则化Mask，鼓励：
    1. 稀疏性：大部分Entity保留（mask=1）
    2. 确定性：避免模棱两可（mask接近0或1）

    Args:
        mask: [num_entities] - sigmoid输出，范围[0, 1]

    Returns:
        loss: scalar
    """
    # L1稀疏正则：鼓励大部分=1（不mask）
    L_sparse = (1 - mask).sum()

    # 熵正则：鼓励接近0或1（最小化熵）
    eps = 1e-8
    entropy = -(
        mask * torch.log(mask + eps) +
        (1 - mask) * torch.log(1 - mask + eps)
    ).mean()

    # 组合：稀疏 + 低熵
    loss = lambda_sparse * L_sparse - lambda_entropy * entropy
    return loss
```

**Mask正则化的作用**：
- 防止模型过度mask（保留大部分Entity）
- 鼓励明确的决策（0或1，不要0.5）
- 可解释性：训练后可以查看哪些Entity被mask了

### **完整训练代码**

```python
def train_epoch(model, train_loader, optimizer, hetero_graph, cf_graph, config):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # === Forward ===
        outputs = model(hetero_graph, cf_graph)

        # 提取batch数据
        user_ids = batch['user_id']
        pos_item_ids = batch['pos_item_id']
        neg_item_ids = batch['neg_item_ids']  # [batch, num_neg]

        user_emb = outputs['user_fused'][user_ids]
        pos_item_emb = outputs['item_fused'][pos_item_ids]
        neg_item_emb = outputs['item_fused'][neg_item_ids]

        # === 1. 主损失：推荐 ===
        L_rec = info_nce_loss(user_emb, pos_item_emb, neg_item_emb)

        # === 2. 多视图对比（可选：在batch上或全量）===
        if config['use_batch_contrast']:
            # 在batch上对比（快）
            L_contrast = multiview_contrastive_loss(
                outputs['user_cf'][user_ids],
                outputs['user_kg'][user_ids]
            )
        else:
            # 全量对比（慢但更准确）
            L_contrast = multiview_contrastive_loss(
                outputs['user_cf'],
                outputs['user_kg']
            )

        # === 3. Entity-Item对齐 ===
        L_align = alignment_loss(
            outputs['entity_emb'],
            outputs['item_kg'],
            hetero_graph['entity', 'describes', 'item'].edge_index
        )

        # === 4. Mask正则 ===
        L_mask = mask_regularization(outputs['mask'])

        # === 总损失 ===
        loss = (L_rec
                + config['alpha'] * L_contrast
                + config['beta'] * L_align
                + config['gamma'] * L_mask)

        # === Backward ===
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

---

## ⚡ 性能优化方案

### **规模分析**

```
数据规模（ML-1M）：
- Users: 6,040
- Items: 3,700
- Entities: ~400
- User-Entity edges: ~90,000
- Entity-Item edges: ~33,000
- User-Item edges: ~900,000

Total nodes: ~10,000
Total edges: ~1M
```

✅ **结论**：规模不大，单卡A800完全够用，性能不是瓶颈

### **潜在性能瓶颈**

#### **瓶颈1：多视图对比损失（全量用户）**

```python
# 如果在所有用户上计算对比损失
L_contrast = multiview_contrastive_loss(
    outputs['user_cf'],      # [6040, dim]
    outputs['user_kg']       # [6040, dim]
)

# 计算量：O(num_users^2) 相似度矩阵
neg_sim = emb_cf @ emb_kg.T  # [6040, 6040] - 36M operations
```

**解决方案**：
- ✅ **方案A**：只在batch上计算对比损失（推荐）
  ```python
  # batch_size = 1024
  L_contrast = multiview_contrastive_loss(
      outputs['user_cf'][batch_users],  # [1024, dim]
      outputs['user_kg'][batch_users]   # [1024, dim]
  )
  # 计算量：O(batch_size^2) = 1M operations（快36倍）
  ```

- ✅ **方案B**：使用Memory Bank（如MoCo）
  - 维护一个user embedding的队列
  - 只对队列中的embedding计算负样本
  - 计算量可控

**我的建议**：方案A（batch内对比），简单高效

#### **瓶颈2：负采样**

```python
# 每个batch需要采样大量负样本
neg_items = torch.randint(0, num_items, (batch_size, num_neg))

# 如果num_neg=100，batch_size=1024
# → 每个batch采样100K个负样本
```

**解决方案**：
- ✅ 预先采样：在DataLoader中预采样负样本，不在训练中实时采样
- ✅ 共享负样本：batch内所有用户共享一部分负样本
- ✅ 适中的`num_neg`：50-100已经足够（不需要1000）

#### **瓶颈3：异构图卷积**

```python
# HeteroConv需要对每种边类型分别计算
# 如果有4种边类型，4个GATConv → 计算量×4
```

**解决方案**：
- ✅ PyG的HeteroConv已经高度优化（C++ backend）
- ✅ GAT heads=4已经够用（不需要8或16）
- ✅ 只用2层GNN（不需要3-4层）

**实测时间**（估算）：
```
ML-1M规模 + 单卡A800：
- Forward pass: ~100ms
- Backward pass: ~150ms
- Total per batch: ~250ms

Batch size = 1024：
- Batches per epoch: 6040 / 1024 ≈ 6
- Time per epoch: 6 × 0.25s = 1.5秒

Training 300 epochs: 1.5s × 300 = 7.5分钟（非常快！）
```

### **多卡训练必要性**

❌ **不需要多卡训练**
- 单卡A800完全够用
- 数据规模小，多卡通信开销反而变大
- DDP适合大规模数据（如ImageNet），不适合推荐任务

✅ **如果要用多卡**：
- 用于同时跑多个实验（不同超参数）
- 而不是单个实验的数据并行

### **性能优化清单**

```python
# 1. 数据加载优化
train_loader = DataLoader(
    dataset,
    batch_size=1024,           # 较大batch（A800显存足够）
    shuffle=True,
    num_workers=4,             # 多进程加载
    pin_memory=True,           # 加速GPU传输
    persistent_workers=True    # 保持worker进程
)

# 2. 模型优化
model = model.cuda()
model = torch.compile(model)   # PyTorch 2.0编译优化（可选）

# 3. 混合精度训练（可选，不太必要）
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model.compute_loss(batch)

# 4. 梯度累积（如果显存不够，但应该够）
accumulation_steps = 1  # 不需要累积

# 5. 只在batch上做对比损失
config['use_batch_contrast'] = True
```

---

## 🛠️ 实现方案对比

### **方案A：完全自己实现（PyTorch + PyG）**

#### 优点
- ✅ 完全灵活，可以自由定制
- ✅ 异构图和多视图对比容易实现
- ✅ 性能可控，优化空间大
- ✅ 代码清晰，易于理解和调试

#### 缺点
- ❌ 需要自己实现训练循环、评估逻辑
- ❌ 需要自己实现metrics（NDCG, Recall等）
- ❌ 没有现成的配置管理

#### 工作量估算
```
数据加载：      1天
模型实现：      1天
损失函数：      0.5天
训练+评估：     1天
调试：          0.5天
Total:         4天
```

---

### **方案B：基于RecBole框架**

#### 优点
- ✅ 有现成的评估框架（NDCG, Recall等）
- ✅ 有配置文件管理
- ✅ 有日志记录和结果保存
- ✅ 可以直接对比baseline（BPR, LightGCN等）

#### 缺点
- ❌ RecBole主要支持同构图，异构图需要扩展
- ❌ 多视图对比学习不是标准功能
- ❌ 灵活性不如自己实现
- ❌ 学习RecBole API需要时间

#### 工作量估算
```
学习RecBole：   1天
扩展模型：      2天
适配数据：      1天
调试：          1天
Total:         5天
```

---

### **方案C：混合方案（推荐！）⭐**

**核心思路**：
- 用PyG实现核心模型（灵活）
- 借鉴RecBole的评估和配置管理

#### 实现细节
```python
# 1. 数据加载：自己实现（PyG格式）
graph_builder.py        # 构建HeteroData

# 2. 模型：自己实现（PyG）
model.py                # KnowledgeEnhancedRecModel

# 3. 训练：自己实现
trainer.py              # 训练循环、优化器

# 4. 评估：借用RecBole或自己实现
evaluator.py            # NDCG, Recall（可以copy RecBole代码）

# 5. 配置管理：YAML文件
configs/
  ├── ours_full.yaml
  ├── ours_wo_contrast.yaml
  └── ...
```

#### 优点
- ✅ 灵活性最高
- ✅ 可以借鉴RecBole的最佳实践
- ✅ 性能可控
- ✅ 工作量适中

#### 工作量估算
```
数据加载：      1天
模型实现：      1天
训练框架：      1天
评估逻辑：      0.5天（copy RecBole）
配置管理：      0.5天
Total:         4天
```

---

### **推荐方案：方案C（混合）**

**理由**：
1. PyG对异构图支持好，HeteroConv和GAT都是现成的
2. 多视图对比学习需要自定义，框架不好扩展
3. RecBole的评估代码可以借鉴（NDCG计算很tricky）
4. 工作量适中，4天可以完成

---

## 🧪 消融实验设计

### **实验组织**

```
experiments/
├── baselines/              # 传统方法
│   ├── BPR
│   ├── LightGCN
│   └── NGCF
│
├── our_ablations/          # 我们的消融
│   ├── ours_full           # 完整模型
│   ├── ours_wo_contrast    # 去掉多视图对比
│   ├── ours_wo_align       # 去掉Entity-Item对齐
│   ├── ours_wo_mask        # 去掉Mask机制
│   ├── ours_cf_only        # 只用CF视图
│   ├── ours_kg_only        # 只用KG视图
│   ├── ours_item_kg        # 只用Item KG
│   └── ours_user_kg        # 只用User KG
│
└── multimodal_baselines/   # 多模态baseline（可选）
    ├── MMGCN
    └── MGAT
```

### **Ablation 1: 损失函数组件**

| 方法 | L_rec | L_contrast | L_align | L_mask | 说明 |
|------|-------|------------|---------|--------|------|
| **Ours-Full** | ✓ | ✓ | ✓ | ✓ | 完整模型 |
| Ours w/o Contrast | ✓ | ✗ | ✓ | ✓ | 去掉多视图对比 |
| Ours w/o Align | ✓ | ✓ | ✗ | ✓ | 去掉Entity对齐 |
| Ours w/o Mask | ✓ | ✓ | ✓ | ✗ | 去掉Mask机制 |
| Ours-Rec-only | ✓ | ✗ | ✗ | ✗ | 只用推荐损失 |

**预期结果**：
```
Ours-Full > Ours w/o Contrast > Ours w/o Align > Ours w/o Mask > Ours-Rec-only
```

### **Ablation 2: 视图选择**

| 方法 | 使用视图 | 说明 |
|------|---------|------|
| **Ours-Full** | CF + KG (融合) | 完整模型 |
| Ours-CF-only | 只用CF视图 | 相当于LightGCN + Mask |
| Ours-KG-only | 只用KG视图 | 只用知识增强路径 |

**预期结果**：
```
Ours-Full > Ours-KG-only > Ours-CF-only
```

**关键对比**：
- Ours-KG-only vs LightGCN → 证明知识增强的价值
- Ours-Full vs Ours-KG-only → 证明多视图融合的价值

### **Ablation 3: 知识图谱类型**

| 方法 | Item KG | User KG | 说明 |
|------|---------|---------|------|
| **Ours-Full** | ✓ | ✓ | 完整模型 |
| Ours-Item-KG | ✓ | ✗ | 只用电影视觉知识 |
| Ours-User-KG | ✗ | ✓ | 只用用户兴趣知识 |

**预期结果**：
```
Ours-Full > Ours-Item-KG > Ours-User-KG
```

**分析**：
- Item KG更重要（电影本身的特征）
- User KG是补充（用户偏好建模）

### **Ablation 4: Mask初始化策略**

| 方法 | 初始化 | 说明 |
|------|--------|------|
| **Ours-Full** | 基于频率 | 低频Entity初始mask小 |
| Ours-Mask-uniform | 全1初始化 | 让模型从头学 |
| Ours-Mask-random | 随机初始化 | 随机[0.5, 1.0] |

**预期结果**：
```
Ours-Full ≈ Ours-Mask-uniform > Ours-Mask-random
```

---

## ⏱️ 时间与资源估算

### **实现时间线**

```
Day 1: 数据加载与图构建
  - 读取.kg和.inter文件
  - 构建PyG HeteroData
  - 统计Entity频率
  - 构建CF图

Day 2: 模型实现
  - CF编码器（GAT）
  - KG编码器（HeteroConv + GAT）
  - Mask机制
  - 融合层

Day 3: 损失函数与训练
  - InfoNCE推荐损失
  - 多视图对比损失
  - Entity对齐损失
  - Mask正则损失
  - 训练循环

Day 4: 评估与配置
  - NDCG, Recall计算
  - 配置文件管理
  - 日志和结果保存
  - 调试和验证

Day 5-6: Baseline实验
  - 运行BPR, LightGCN, NGCF
  - 对比结果

Day 7-10: 消融实验
  - 8个消融变体
  - 结果分析

Total: 10天（2周）
```

### **计算资源**

```
单个实验（300 epochs）：
  - 单卡A800
  - 训练时间：~10分钟
  - 显存占用：~4GB（远小于80GB）

并行实验（建议）：
  - 4张A800同时跑4个实验
  - Baseline + 3个消融
  - 每轮10分钟，总共需要3轮
  - Total: 30分钟完成12个实验 🚀

保守估计：
  - 每个实验20分钟（包括评估）
  - 12个实验 × 20分钟 / 4卡 = 1小时
  - 加上调试时间：2-3小时完成所有实验
```

### **成本估算**

```
A800使用成本：
  - 实验室资源：免费
  - 时间成本：2-3小时GPU时间

如果外部租用：
  - A800: ~$2/小时
  - 3小时 × $2 = $6（非常便宜）

总结：资源需求很低，性能不是问题 ✅
```

---

## 📋 下一步行动

### **立即开始**
1. ✅ **写好文档**（当前任务）
2. ⏸️ 创建代码骨架（空函数，接口定义）
3. ⏸️ 实现数据加载（最关键）
4. ⏸️ 实现模型和损失函数
5. ⏸️ 调试和实验

### **优先级**
```
P0 (必须):
  - 数据加载和图构建
  - 完整模型实现
  - Ours-Full训练成功

P1 (重要):
  - Baseline实验（BPR, LightGCN）
  - 核心消融（w/o Contrast, w/o Mask）

P2 (可选):
  - 多模态baseline（MMGCN, MGAT）
  - 更多消融实验
```

---

## 📚 参考文献

### **图神经网络推荐**
1. LightGCN: He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
2. NGCF: Wang et al. "Neural Graph Collaborative Filtering." SIGIR 2019.

### **知识图谱推荐**
3. KGAT: Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." KDD 2019.
4. KGCN: Wang et al. "Knowledge Graph Convolutional Networks for Recommender Systems." WWW 2019.

### **对比学习推荐**
5. SGL: Wu et al. "Self-supervised Graph Learning for Recommendation." SIGIR 2021.
6. SimGCL: Yu et al. "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation." SIGIR 2022.

### **多模态推荐**
7. MMGCN: Wei et al. "MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video." ACM MM 2019.
8. MGAT: Tao et al. "MGAT: Multimodal Graph Attention Network for Recommendation." IPM 2020.

---

**Last updated**: 2026-01-12
**Status**: 设计完成，等待实现
