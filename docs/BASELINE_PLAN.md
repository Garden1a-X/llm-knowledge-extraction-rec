# Baseline 实验规划

## 研究定位

**你们的方法**：多模态知识图谱推荐（Multimodal Knowledge Graph Enhanced Recommendation）

## Baseline 分类体系

### 1️⃣ 传统推荐方法（无图谱，无多模态）

| 方法 | 说明 | 目的 |
|------|------|------|
| **BPR** | 协同过滤，矩阵分解 | 证明图谱和多模态的必要性 |

**对比意义**：作为最基础的baseline，证明引入图谱或多模态的价值。

---

### 2️⃣ 图谱推荐方法（有图谱，无多模态）

| 方法 | 类型 | 图谱结构 | 说明 |
|------|------|----------|------|
| **LightGCN** | GCN | User-Item 二部图 | 最简单的图卷积 |
| **NGCF** | GCN | User-Item 二部图 | 带消息传递的图卷积 |
| **KGAT** | GAT | User-Item-Entity 异构图 | 知识图谱注意力网络 |

**对比意义**：
- 证明图谱结构的价值
- 对比不同图谱架构（GCN vs GAT）
- 这些方法用**标准图谱**（user-item交互）

---

### 3️⃣ 多模态推荐方法（有多模态，无/简单图谱）

| 方法 | 模态 | 说明 | 建议 |
|------|------|------|------|
| **VBPR** | Visual | 视觉特征 + MF | 经典视觉推荐 |
| **DeepStyle** | Visual | 视觉风格 + CF | 风格感知推荐 |
| **ACF** | Visual | 注意力机制 + 视觉特征 | 注意力感知推荐 |

**对比意义**：
- 证明多模态信息的价值
- 但这些方法**没有使用图谱结构**
- 或只用简单的图谱（非知识图谱）

---

### 4️⃣ 多模态图谱推荐方法（有多模态 + 有图谱）⭐

| 方法 | 模态 | 图谱 | 说明 | 优先级 |
|------|------|------|------|--------|
| **MMGCN** | V+A+T | 多模态异构图 | 多模态图卷积 | ⭐⭐⭐ 最相关 |
| **MGAT** | V+A+T | 多模态异构图 | 多模态图注意力 | ⭐⭐⭐ 最相关 |
| **GRCN** | V+A+T | 多模态图 | 图精炼卷积 | ⭐⭐ |
| **LATTICE** | V+A+T | 多模态KG | 多模态知识图谱 | ⭐⭐⭐ 很相关 |

**对比意义**：
- ⭐ **最核心的对比组**
- 这些方法与你们的工作**最相似**
- 用于证明你们的知识增强图谱比标准多模态图谱更好

---

### 5️⃣ LLM推荐方法（可选，展示最新进展）

| 方法 | 说明 | 建议 |
|------|------|------|
| **LLM4Rec** | LLM作为推荐器 | 可选 |
| **TALLRec** | LLM + 推荐 | 可选 |
| **RecLLM** | 推荐系统 + LLM | 可选 |

---

## 🎯 精简后的 Baseline 组合（共12个方法）

### **1. 传统推荐方法** (1个)
- ✅ **BPR** - 协同过滤 baseline

### **2. 图谱推荐方法** (2个)
- ✅ **LightGCN** - 简化图卷积，SOTA图谱方法
- ✅ **KGAT** - 知识图谱注意力网络

~~删除：NGCF（与LightGCN类似）~~

### **3. 多模态推荐方法**（无/简单图谱）(2个)
- 🔍 **VBPR** - 视觉贝叶斯个性化排序（经典）
- 🔍 **ACF** 或 **DeepStyle** - 注意力/风格感知推荐

### **4. LLM推荐方法** (2个)
- 🔍 **LLM4Rec** - LLM作为推荐器
- 🔍 **TALLRec** 或 **RecLLM** - LLM增强推荐

### **5. 多模态图谱方法** (3个) ⭐ 最相关
- 🔍 **MMGCN** - 多模态图卷积网络
- 🔍 **MGAT** - 多模态图注意力网络
- 🔍 **LATTICE** 或 **GRCN** - 另一个多模态图谱方法

### **6. MLLM推荐方法**（多模态LLM）(2个) 🆕
- 🔍 **VIP5** - 视觉指令微调推荐
- 🔍 **LlamaRec** 或 **InstructRec** - 指令微调MLLM推荐

### **7. 你们的方法** (1个)
- 💡 **Ours-Full** - 多模态知识图谱推荐（LLM知识增强）

---

**图例**：
- ✅ = 已有RecBole实现，可直接运行
- 🔍 = 需要找开源实现或自己实现
- 💡 = 你们的创新方法

---

## 📊 对比逻辑链

### **演进路径 1：从传统到多模态图谱**
```
BPR (传统CF)
  ↓ (+图谱结构)
LightGCN/KGAT (图谱推荐)
  ↓ (+多模态信息)
MMGCN/MGAT/LATTICE (多模态图谱，标准多模态特征)
  ↓ (+LLM提取的知识)
Ours-Full (多模态知识图谱，LLM知识增强) ← 你们的方法 💡
```

### **演进路径 2：从多模态到知识增强**
```
BPR (传统CF)
  ↓ (+视觉特征)
VBPR/ACF (多模态推荐)
  ↓ (+图谱结构)
MMGCN/MGAT (多模态图谱)
  ↓ (+LLM知识点)
Ours-Full ← 你们的方法 💡
```

### **演进路径 3：从LLM到知识图谱**
```
LLM4Rec/TALLRec (LLM推荐)
  ↓ (+多模态)
VIP5/LlamaRec (MLLM推荐)
  ↓ (+知识图谱结构)
Ours-Full ← 你们的方法 💡
```

### **核心对比**
```
多模态图谱方法组 (MMGCN, MGAT, LATTICE)
     vs
你们的方法 (Ours-Full)
     ↓
证明：LLM提取的细粒度知识点 > 标准多模态特征
```

---

## 🔬 消融实验（Ablation Studies）

### **消融1：知识增强图谱的有效性**

| 方法 | 图谱 | 结果预期 |
|------|------|----------|
| MMGCN + 标准图谱 | User-Item-Modality | Baseline |
| MMGCN + Ours图谱 | User-Knowledge-Item | **提升** ⬆️ |

**证明**：LLM提取的知识点构建的图谱 > 标准多模态图谱

### **消融2：LLM能力的影响**

| 方法 | 知识提取器 | 结果预期 |
|------|-----------|----------|
| Ours + Qwen3-VL-8B | 开源VLM | Baseline |
| Ours + GPT-4o-mini | 闭源VLM | **提升** ⬆️ |

**证明**：更强的LLM提取的知识 → 更好的推荐效果

---

## 📁 实验配置

### 数据预处理

- **过滤**：5-core filtering（与MMGCN/MGAT对齐）
- **数据集**：MovieLens-1M
- **分割**：70% train / 10% val / 20% test
- **排序**：Time-based ordering (TO)

### 实验设置

- **Trials**：5次（不同随机种子）
- **Epochs**：300
- **Seeds**：[42, 2023, 2024, 2025, 12345]
- **Metrics**：NDCG@10, Recall@10, Precision@10, Hit@10

### 评估协议

- **Ranking mode**：Full ranking
- **Metric**：NDCG@10 (主要), Recall@10, Precision@10

---

## 📅 实验时间线

### Phase 1: 传统和图谱方法（1-2天）
- [ ] BPR (5 trials × 300 epochs)
- [ ] LightGCN (5 trials × 300 epochs)
- [ ] NGCF (5 trials × 300 epochs)

### Phase 2: 多模态方法（需要实现，3-5天）
- [ ] VBPR - 需要提取视觉特征
- [ ] 或跳过，直接对比多模态图谱方法

### Phase 3: 多模态图谱方法（需要实现，5-7天）
- [ ] MMGCN - 需要实现或找开源代码
- [ ] MGAT - 需要实现或找开源代码

### Phase 4: 你们的方法（核心工作，2-3周）
- [ ] 实现Ours模型
- [ ] 消融实验

---

## 💡 建议

1. **立即开始**：
   - ✅ 添加 5-core 过滤
   - ✅ 用新数据重跑 BPR, LightGCN, NGCF

2. **MMGCN/MGAT 实现**：
   - 查找开源代码
   - 或者联系作者要代码
   - 或者参考论文自己实现

3. **VBPR 可选**：
   - 如果时间紧，可以跳过单纯的多模态方法
   - 直接对比多模态图谱方法（MMGCN/MGAT）

4. **优先级**：
   ```
   BPR > LightGCN > NGCF > MMGCN/MGAT > VBPR > Others
   ```

---

## 📚 参考文献

1. **BPR**: Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
2. **LightGCN**: He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.
3. **NGCF**: Wang et al. Neural Graph Collaborative Filtering. SIGIR 2019.
4. **KGAT**: Wang et al. KGAT: Knowledge Graph Attention Network for Recommendation. KDD 2019.
5. **VBPR**: He & McAuley. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. AAAI 2016.
6. **MMGCN**: Wei et al. MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. ACM MM 2019.
7. **MGAT**: Tao et al. MGAT: Multimodal Graph Attention Network for Recommendation. IPM 2020.
8. **LATTICE**: Zhang et al. LATTICE: Mining Latent Type Information for Improving Recommendation with KGs. SIGIR 2021.

---

*Last updated: 2025-12-27*
