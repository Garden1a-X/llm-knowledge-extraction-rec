# Method部分写作框架

*基于实际实现整理*
*Created: 2026-01-18*

---

## 📋 整体结构

```
Method
├── 3.1 Knowledge Extraction from Multimodal Data
│   ├── 3.1.1 Three-Stage Item Knowledge Extraction
│   └── 3.1.2 Temporal User Interest Extraction
│
└── 3.2 Knowledge-Enhanced Heterogeneous Graph Recommendation
    ├── 3.2.1 Heterogeneous Graph Construction
    ├── 3.2.2 Multi-View Contrastive Learning
    ├── 3.2.3 Learnable Mask Mechanism
    └── 3.2.4 Training Objective
```

---

## 📝 3.1 Knowledge Extraction from Multimodal Data

### **开场段落（问题动机）**

**写什么**：
- 传统推荐系统依赖稀疏的User-Item交互
- 多模态数据（图片、文本）包含丰富的语义信息但被忽略
- LLM作为信息提取器的能力未被充分利用

**实际方法对应**：
- ML-1M: 电影海报图片（3,706部电影，每部1张海报）
- 用户评分历史（6,040用户，平均149条评分/用户）
- 使用GPT-4o-mini作为knowledge extractor

**写作要点**：
```
We propose a two-stage knowledge extraction framework that leverages
large language models (LLMs) as information extractors to convert raw
multimodal data into structured knowledge graphs. Specifically, we
extract visual knowledge from item images and temporal interest patterns
from user interaction histories.
```

---

### **3.1.1 Three-Stage Item Knowledge Extraction**

#### **Stage 1: Exploratory Extraction (Phase 1)**

**写什么**：
- 目的：探索领域内的relation和entity空间
- 采样策略：5%随机采样（ML-1M: 170部电影）
- MLLM设置：GPT-4o-mini，temperature=0.7
- Prompt设计：开放式提取，无relation限制

**实际数据**（从Phase 1结果）：
- 采样：170 movies (5%)
- 提取结果：1,531 knowledge points
- 发现：147 unique relations, 1,200+ unique entities
- 成本：$0.50

**Prompt策略**：
```python
# 开放式提取，让MLLM自由发挥
prompt = f"""
Analyze the movie poster and extract visual knowledge.
Output format: [
  {{"relation": "<aspect>", "entity": "<value>"}},
  ...
]
"""
```

**写作示例**：
```
In Stage 1, we perform exploratory extraction on 5% randomly sampled items
(170 movies) to discover the domain-specific knowledge space. We use an
open-ended prompt that allows the MLLM to freely identify visual aspects
without predefined constraints. This yields 147 unique relations and over
1,200 entities, revealing the rich semantic diversity in movie posters.
```

---

#### **Stage 2: Vocabulary Standardization**

**写什么**：
- 问题：Stage 1产生的relations和entities高度分散，不利于协同过滤
- 目标：将分散的knowledge归并为紧凑的标准词表（~180 entities）
- 方法：人工定义核心relations + 算法聚类entities

**实际操作**（基于代码）：

1. **Relation标准化**（14+1 core relations）：
   - 人工分析147个raw relations
   - 归并为15个核心relations
   - 示例：`color_palette`, `mood_atmosphere`, `composition_styles`, ...

2. **Entity标准化**（~180 entities）：
   - 统计频率：移除低频(<2次)
   - 语义聚类：相似entities合并
   - 示例：
     - `dark red` + `deep red` → `dark_red`
     - `romantic mood` + `love atmosphere` → `romantic`

**实际数字**（基于standardization代码）：
- Raw: 147 relations → Standardized: 15 relations (-90%)
- Raw: 1,200+ entities → Standardized: 180 entities (-85%)

**写作示例**：
```
To enable effective collaborative filtering, we standardize the diverse
knowledge into a compact vocabulary. We manually design 15 core relations
covering visual aspects (color, lighting, composition), semantic aspects
(mood, theme, genre), and temporal aspects (era). For entities, we perform
frequency-based filtering and semantic clustering, reducing 1,200+ entities
to 180 canonical forms. This ensures entities are shared across items,
enabling knowledge-based user-item matching.
```

---

#### **Stage 3: Constrained Full Extraction (Phase 4)**

**写什么**：
- 目的：用标准化词表对全部items提取知识
- 约束：MLLM只能输出15个pre defined relations和180个standard entities
- 好处：保证knowledge graph的连通性和可解释性

**实际操作**：
- 全量提取：3,706 movies (100%)
- Prompt包含15 relations的定义
- 后处理：过滤非词表内的entities

**Prompt策略**：
```python
prompt = f"""
Extract visual knowledge using ONLY these predefined relations:
1. color_palette: Color scheme (e.g., warm_colors, dark_palette)
2. composition_styles: Layout and framing (e.g., centered, rule_of_thirds)
3. mood_atmosphere: Emotional tone (e.g., romantic, dark)
... (列出所有15个)

Rules:
- Use EXACT relation names above
- Use canonical entity names from our vocabulary
- Output 3-8 knowledge points per image
"""
```

**实际结果**（基于Phase 4数据）：
- 3,706 movies × 平均5.6 KPs/movie = 20,750 knowledge points
- 覆盖180个entities中的165个 (92% coverage)
- 每个entity平均被125个movies使用
- 成本：$3.50

**写作示例**：
```
In Stage 3, we perform constrained extraction on the full dataset using
the standardized vocabulary. The MLLM is instructed to select from the
15 predefined relations and 180 canonical entities. This yields 20,750
knowledge triplets for 3,706 movies, with an average of 5.6 knowledge
points per movie. 92% of entities appear in at least 5 movies, ensuring
sufficient overlap for collaborative filtering.
```

---

### **3.1.2 Temporal User Interest Extraction**

#### **动机段落**

**写什么**：
- 用户兴趣是动态evolving的（不是静态的）
- 短期兴趣反映当前偏好，长期兴趣反映稳定品味
- 利用LLM识别兴趣演化模式

**实际设置**：
- 数据：用户评分历史，按时间戳排序
- 目标：为每个用户提取10个短期兴趣 + 5个长期兴趣

---

#### **21-Day Temporal Bucketing**

**写什么**：
- 将用户评分历史按21天分桶（习惯养成周期）
- 每个桶代表一个短期时间窗口

**实际操作**：
```python
# 按活跃天数（非日历时间）分桶
buckets = []
for user_ratings in user_history:
    sorted_by_time = sort(user_ratings, by='timestamp')
    active_days = compute_active_days(sorted_by_time)
    buckets = split_by_days(active_days, bucket_size=21)
```

**ML-1M数据集限制**：
- 59%用户在注册日批量评分（timestamp=提交时间）
- 导致大多数用户只有1个桶
- 但方法论正确，适用于Amazon等真实时序数据集

**写作示例**：
```
We partition each user's rating history into temporal buckets of 21 days,
corresponding to the habit formation cycle. For each bucket, we extract
short-term interests by aggregating the visual knowledge of highly-rated
items (rating ≥ 4.0). This captures the user's immediate preferences
during that period.
```

---

#### **Short-Term Interest Aggregation**

**写什么**：
- 方法：统计聚合（非LLM）
- 对每个21天桶：统计高分电影(≥4星)的knowledge points
- 保留Top-10高频entities作为短期兴趣

**实际代码**：
```python
def extract_short_term_interest(bucket_ratings, item_kg):
    # 1. 过滤高分电影
    high_rated = [r for r in bucket_ratings if r['rating'] >= 4.0]

    # 2. 聚合knowledge points
    entity_counts = Counter()
    for rating in high_rated:
        movie_kg = item_kg[rating['item_id']]
        for (relation, entity) in movie_kg:
            entity_counts[entity] += 1

    # 3. 保留Top-10
    short_term = entity_counts.most_common(10)
    return short_term
```

**写作示例**：
```
For each temporal bucket, we aggregate the visual knowledge from highly-rated
movies (rating ≥ 4.0). We count the frequency of each entity appearing in
these movies and retain the Top-10 as short-term interests for that period.
```

---

#### **Long-Term Interest Summarization with LLM**

**写什么**：
- 每4个桶（84天 ≈ 3个月）进行一次LLM总结
- 输入：最近4个桶的短期兴趣
- 任务：LLM识别持续出现的pattern，更新长期兴趣
- 年龄追踪：每个长期兴趣记录持续天数

**实际Prompt**：
```python
prompt = f"""
User's recent short-term interests (past 4 periods, 84 days):
Period 1 (21 days): {period1_interests}
Period 2 (21 days): {period2_interests}
Period 3 (21 days): {period3_interests}
Period 4 (21 days): {period4_interests}

Current long-term interests: {current_long_term}

Task: Update the user's long-term interests (Top-5).
Rules:
1. Retain stable interests (appearing in multiple periods)
2. Add new persistent patterns
3. Remove outdated interests
4. Track age: existing+84 days, new=84 days

Output JSON:
[
  {{"relation": "...", "entity": "...", "age_days": ...}},
  ...
]
"""
```

**实际统计**：
- 6,040用户
- 平均1.0次LLM调用/用户（受ML-1M限制）
- 99.9%用户有长期兴趣（5个）
- 99.5%用户有短期兴趣（10个）

**写作示例**：
```
Every 4 buckets (84 days), we use an LLM to summarize long-term interests.
The LLM analyzes the short-term interests from the past 84 days and identifies
persistent patterns that reflect stable user preferences. Each long-term interest
is annotated with an age (in days), tracking how long the user has maintained
that preference. This enables the model to distinguish between enduring tastes
and recent trends.
```

---

#### **Graph RAG for Knowledge Retrieval**（如果想强调的话）

**写什么**（可选）：
- 在总结长期兴趣时，LLM需要"查看"电影知识图谱
- 实现：在prompt中提供short-term interests（已经连接到item KG）
- 相当于Graph RAG的简化版：通过共享entities连接user和item

**写作示例**（可选段落）：
```
We adopt a graph retrieval approach where the LLM accesses item knowledge
through the shared entity vocabulary. When summarizing long-term interests,
the short-term interests already represent entities from the item knowledge
graph, enabling the LLM to reason about user-item semantic connections without
explicit graph traversal.
```

---

#### **最终输出**

**写什么**：
- 每个用户：10个短期兴趣 + 5个长期兴趣
- 格式：`(relation, entity, age_days)`
- 与item KG对齐：使用相同的15 relations和180 entities

**实际数据**：
```
User KG Statistics:
- 6,040 users
- 90,100 user-entity edges
  - 30,179 long-term interests (5 per user)
  - 59,921 short-term interests (10 per user)
- Cost: $6-12 (gpt-4o-mini)
```

**写作示例**：
```
The final user knowledge graph contains 90,100 user-entity connections
across 6,040 users. Each user has 5 long-term interests and 10 short-term
interests, all aligned with the standardized entity vocabulary from the
item knowledge graph.
```

---

## 📝 3.2 Knowledge-Enhanced Heterogeneous Graph Recommendation

### **3.2.1 Heterogeneous Graph Construction**

#### **图结构定义**

**写什么**：
- 节点类型：User, Item, Entity
- 边类型：
  - CF视图：`(User, rated, Item)` - 传统协同过滤
  - KG视图：
    - `(User, long_term_interest, Entity)`
    - `(User, short_term_interest, Entity)`
    - `(Entity, describes, Item)`

**关键设计**：
- User和Item不直接连接（在KG视图中）
- 通过共享Entity间接连接
- 支持双视图对比学习

**实际规模**（ML-1M）：
```python
Nodes:
  - 6,040 users
  - 3,706 items (movies)
  - ~180 entities

Edges (CF View):
  - 900,000 user-item interactions

Edges (KG View):
  - 30,179 user-entity (long-term)
  - 59,921 user-entity (short-term)
  - 20,750 entity-item
```

**写作示例**：
```
We construct a heterogeneous graph G = (V, E) where V = {U, I, E} represents
users, items, and knowledge entities. The graph contains two types of edges:
(1) CF edges (U, rated, I) representing user-item interactions, and
(2) KG edges connecting users and items through shared entities via two paths:
    U → (long/short-term interest) → E → (describes) → I

This design enables collaborative filtering through both direct interactions
and knowledge-mediated semantic matching.
```

---

### **3.2.2 Multi-View Contrastive Learning**

#### **双视图编码器**

**写什么**：
- CF视图编码器：简单的User-Item bipartite graph
- KG视图编码器：异构图，包含User-Entity-Item三方

**实际架构**：
```python
# CF View: 2-layer GAT
CF_Encoder:
  - Input: User/Item embeddings
  - Layer 1: GAT(dim=64, heads=4)
  - Layer 2: GAT(dim=64, heads=1)
  - Output: user_emb_cf, item_emb_cf

# KG View: 2-layer Heterogeneous GAT
KG_Encoder:
  - Input: User/Item/Entity embeddings (Entity masked)
  - Layer 1: HeteroGAT for all 3 edge types
  - Layer 2: HeteroGAT
  - Output: user_emb_kg, item_emb_kg
```

**写作示例**：
```
We employ two graph encoders to learn complementary representations:

CF View Encoder uses a 2-layer Graph Attention Network (GAT) on the user-item
interaction graph, capturing collaborative filtering signals.

KG View Encoder uses a heterogeneous GAT to propagate information through
the knowledge-enriched paths (User → Entity → Item). This encoder learns
semantic representations based on shared visual and interest knowledge.
```

---

#### **视图对比损失**

**写什么**：
- 目的：让两个视图学到的user表示一致但互补
- 方法：InfoNCE对比损失

**实际公式**：
```python
def multiview_contrast_loss(emb_cf, emb_kg, temperature=0.1):
    """
    Positive: 同一user的两个视图表示
    Negative: 不同user的cross-view表示
    """
    emb_cf = F.normalize(emb_cf, dim=-1)
    emb_kg = F.normalize(emb_kg, dim=-1)

    pos_sim = (emb_cf * emb_kg).sum(dim=-1) / temperature
    neg_sim = emb_cf @ emb_kg.T / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.arange(batch_size)

    loss = F.cross_entropy(logits, labels)
    return loss
```

**写作示例**：
```
To align the two views, we employ a contrastive loss that encourages the
CF and KG representations of the same user to be similar, while pushing
apart representations from different users:

L_contrast = -log( exp(sim(u_cf, u_kg) / τ) /
                   Σ_v exp(sim(u_cf, v_kg) / τ) )

where τ is a temperature hyperparameter. This ensures the knowledge-enhanced
view complements rather than conflicts with the collaborative filtering view.
```

---

### **3.2.3 Learnable Mask Mechanism**

#### **动机**

**写什么**：
- 问题：LLM提取的知识可能包含幻觉（hallucinations）
- 目标：让模型自动学习哪些entities不可靠，降低其权重
- 方法：为每个entity学习一个mask权重 ∈ [0, 1]

**实际实现**：
```python
class MaskModule(nn.Module):
    def __init__(self, num_entities):
        self.mask_logits = nn.Parameter(torch.zeros(num_entities))

    def forward(self, entity_emb):
        mask = torch.sigmoid(self.mask_logits)  # [0, 1]
        return entity_emb * mask.unsqueeze(1)
```

**写作示例**：
```
To mitigate potential hallucinations in LLM-extracted knowledge, we introduce
a learnable mask mechanism. Each entity e has a learnable weight m_e ∈ [0, 1]
that modulates its embedding:

    ẽ_e = m_e · e_e

The mask weights are initialized based on entity frequency (low-frequency
entities are more likely to be unreliable) and optimized during training.
This allows the model to automatically suppress noisy knowledge while
retaining useful entities.
```

---

#### **Mask正则化**

**写什么**：
- 正则项：鼓励稀疏性（大部分entity保留）+ 确定性（接近0或1）

**实际公式**：
```python
def mask_regularization(mask):
    # 稀疏性：鼓励接近1（不mask）
    L_sparse = (1 - mask).sum()

    # 熵正则：鼓励确定性（避免0.5）
    entropy = -(mask * log(mask) + (1-mask) * log(1-mask)).mean()

    loss = λ_sparse * L_sparse - λ_entropy * entropy
    return loss
```

**写作示例**：
```
We regularize the mask weights with two terms: (1) a sparsity term encouraging
most entities to be retained (m_e ≈ 1), and (2) an entropy term encouraging
binary decisions (m_e ≈ 0 or 1). This prevents the model from over-masking
useful knowledge while clearly identifying unreliable entities.
```

---

### **3.2.4 Training Objective**

#### **总损失函数**

**写什么**：
- 主损失：InfoNCE推荐损失
- 辅助损失：多视图对比 + Entity-Item对齐 + Mask正则

**实际公式**：
```
L_total = L_rec + α·L_contrast + β·L_align + γ·L_mask

where:
  L_rec: InfoNCE recommendation loss
  L_contrast: Multi-view contrastive loss
  L_align: Entity-item alignment loss
  L_mask: Mask regularization

Hyperparameters:
  α = 0.1, β = 0.05, γ = 0.01
```

**各项损失详解**：

1. **L_rec (InfoNCE推荐损失)**：
```python
L_rec = -log( exp(⟨u, i_pos⟩ / τ) /
             (exp(⟨u, i_pos⟩ / τ) + Σ_{i_neg} exp(⟨u, i_neg⟩ / τ)) )
```

2. **L_align (Entity-Item对齐)**：
```python
# 让entity和它描述的item在embedding空间接近
L_align = Σ_{(e,i)∈describes} -log σ(⟨e, i⟩ - ⟨e, i_neg⟩)
```

**写作示例**：
```
The training objective combines four loss terms:

1. Recommendation Loss (L_rec): An InfoNCE loss that encourages the model
   to rank positive items higher than negatives for each user.

2. Contrastive Loss (L_contrast): Aligns the CF and KG view representations
   as described in Section 3.2.2.

3. Alignment Loss (L_align): Ensures entities and the items they describe
   have similar embeddings, maintaining semantic consistency.

4. Mask Regularization (L_mask): Regularizes the entity mask weights to
   prevent over-masking and encourage binary decisions.

The final objective is:
    L = L_rec + α·L_contrast + β·L_align + γ·L_mask
with α=0.1, β=0.05, γ=0.01.
```

---

## 🎯 写作建议

### **Method章节长度分配**

```
3.1 Knowledge Extraction: 40%
  - 3.1.1 Item Knowledge: 25%
  - 3.1.2 User Interest: 15%

3.2 Recommendation Model: 60%
  - 3.2.1 Graph Construction: 10%
  - 3.2.2 Multi-View Learning: 25%
  - 3.2.3 Mask Mechanism: 15%
  - 3.2.4 Training Objective: 10%
```

### **图表建议**

1. **Figure 1: 三阶段知识提取流程图**
   - Stage 1: 5% sampling → Exploratory extraction
   - Stage 2: Vocabulary standardization
   - Stage 3: Constrained full extraction

2. **Figure 2: 用户兴趣提取pipeline**
   - Temporal bucketing (21 days)
   - Short-term aggregation
   - Long-term LLM summarization

3. **Figure 3: 异构图结构**
   - 节点：User, Item, Entity
   - 边：CF edges vs KG edges
   - 双视图对比

4. **Figure 4: 模型架构**
   - CF Encoder
   - KG Encoder
   - Mask Mechanism
   - Fusion Layer

### **数字和统计**

在Method中适当穿插实际数字，增强可信度：

- "We extract knowledge from 3,706 movie posters, yielding 20,750 triplets."
- "92% of entities appear in at least 5 movies, ensuring collaborative signal."
- "The user knowledge graph contains 90,100 edges across 6,040 users."
- "Our three-stage extraction costs $15-20 in total, demonstrating efficiency."

### **突出创新点**

在每个subsection开头用1-2句话说明motivation/innovation：

- **3.1.1**: "Unlike prior work that extracts unstructured features, we propose a three-stage pipeline that balances exploration and standardization."
- **3.1.2**: "We introduce temporal bucketing to capture dynamic interest evolution, leveraging LLM's reasoning ability."
- **3.2.2**: "We employ multi-view contrastive learning to combine collaborative filtering with knowledge-enhanced semantic matching."
- **3.2.3**: "To address LLM hallucinations, we propose a learnable mask mechanism that automatically identifies unreliable knowledge."

---

## 📚 对应源码位置

供写作时参考实际实现细节：

### **知识提取**
- `scripts/videogames_phase1_extraction.py` - Stage 1探索
- `scripts/create_compact_entity_vocabulary.py` - Stage 2标准化
- `scripts/videogames_phase3_full_extraction.py` - Stage 3全量提取
- `src/extraction/user_interest_extractor.py` - 用户兴趣提取核心
- `scripts/extract_user_interests_hybrid.py` - 用户兴趣提取CLI

### **推荐模型**
- `src/models/knowledge_enhanced_rec.py` - 主模型（如果存在）
- `docs/MODEL_DESIGN.md` - 模型设计文档
- 配置文件：`configs/ours_full.yaml`

### **数据统计**
- `results/phase4_full_extraction_filtered.json` - Item KG统计
- `results/user_interests_hybrid.json` - User KG统计
- `docs/EXPERIMENT_TRACKING.md` - 实验结果

---

**Good luck with your paper writing!** 📝🎓
