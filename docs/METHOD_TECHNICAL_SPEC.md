# Knowledge-Enhanced Recommendation Model: Technical Specification

> **目标读者**: 论文撰写者
> **用途**: Method Section (Section 3.2) 的技术支撑文档
> **抽象层级**: 高层设计思想 + 数学公式，**不包含**具体超参数和实现细节

---

## 1. 异构图构建 (Heterogeneous Graph Construction)

### 1.1 图的符号定义

我们构建一个知识增强的异构图：

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E})
$$

其中节点集合 $\mathcal{V}$ 包含三种类型：

$$
\mathcal{V} = \mathcal{U} \cup \mathcal{I} \cup \mathcal{E}
$$

- $\mathcal{U}$: 用户节点集合（User nodes）
- $\mathcal{I}$: 物品节点集合（Item nodes）
- $\mathcal{E}$: 知识实体节点集合（Knowledge entity nodes），由LLM从电影海报中提取

### 1.2 边的类型与语义

边集合 $\mathcal{E}$ 包含三种类型的语义边（及其反向边）：

#### (1) User-Entity 边（用户兴趣）

我们区分两种时间尺度的用户兴趣：

- **长期兴趣** (Long-term interests): $\mathcal{E}_{\text{long}} = \{(u, e) \mid u \in \mathcal{U}, e \in \mathcal{E}\}$
  - 语义：用户 $u$ 对知识实体 $e$ 有**持久的**偏好
  - 来源：基于用户历史观影的TF-IDF加权知识聚合（高频+独特性）

- **短期兴趣** (Short-term interests): $\mathcal{E}_{\text{short}} = \{(u, e) \mid u \in \mathcal{U}, e \in \mathcal{E}\}$
  - 语义：用户 $u$ 对知识实体 $e$ 有**近期的**偏好
  - 来源：基于用户最近观影的知识点频率（高频但不一定独特）

**设计动机**：长期兴趣捕获用户稳定的视觉品味（如"喜欢noir风格"），短期兴趣捕获用户当前的探索趋势（如"最近在看科幻片"）。两种边类型在GNN中使用不同的参数独立学习。

#### (2) Entity-Item 边（知识描述）

$$
\mathcal{E}_{\text{desc}} = \{(e, i) \mid e \in \mathcal{E}, i \in \mathcal{I}\}
$$

- 语义：知识实体 $e$ **描述**了物品 $i$ 的视觉特征
- 来源：LLM从电影海报提取的三元组 $\langle \text{item}, \text{relation}, \text{entity} \rangle$
- **注意**：图中**不保留**具体的relation类型（如`color_palette`, `mood`等），只保留entity本身。Relation信息在知识提取阶段用于组织entity，但在图结构中被抽象为统一的"describes"语义。

**设计动机**：Entity作为语义桥梁连接User和Item，使得"喜欢noir风格的用户"可以通过"noir"这个entity发现"具有noir风格的电影"。

#### (3) User-Item 边（隐式反馈）

$$
\mathcal{E}_{\text{inter}} = \{(u, i) \mid u \in \mathcal{U}, i \in \mathcal{I}\}
$$

- 语义：用户 $u$ 与物品 $i$ 有**交互**（观看/评分）
- 来源：隐式反馈数据（评分 $\geq 4.0$ 视为正样本）
- **注意**：此边仅用于CF视图，不在KG视图中使用

#### (4) 反向边 (Reverse edges)

为确保异构GNN中所有节点类型都能被更新，每种边都有对应的反向边：

- $\mathcal{E}_{\text{rev\_long}}$, $\mathcal{E}_{\text{rev\_short}}$: Entity → User
- $\mathcal{E}_{\text{rev\_desc}}$: Item → Entity

**设计动机**：在异构图中，单向边会导致某些节点类型只接收信息而不发送信息（或相反），反向边确保信息双向流动。

### 1.3 完整的边集合

$$
\mathcal{E} = \mathcal{E}_{\text{long}} \cup \mathcal{E}_{\text{short}} \cup \mathcal{E}_{\text{desc}} \cup \mathcal{E}_{\text{inter}} \cup \mathcal{E}_{\text{rev\_long}} \cup \mathcal{E}_{\text{rev\_short}} \cup \mathcal{E}_{\text{rev\_desc}}
$$

---

## 2. 双视图表示学习 (Dual-View Representation Learning)

我们的模型使用**两个独立的GNN编码器**从不同视角学习用户和物品的表示：

- **CF View**: 捕获协同过滤信号（"相似用户喜欢相似物品"）
- **KG View**: 捕获知识增强信号（"用户通过entity偏好发现物品"）

### 2.1 CF View (Collaborative Filtering View)

#### GNN架构选择

**使用GAT (Graph Attention Network)**，原因：
- 自适应聚合邻居信息（不同邻居贡献不同）
- 适合推荐场景中的长尾问题（热门物品权重自动降低）

#### 编码过程

CF视图在User-Item二部图上进行消息传递。设初始embedding为：

$$
\mathbf{h}_u^{(0)} = \mathbf{e}_u, \quad \mathbf{h}_i^{(0)} = \mathbf{e}_i
$$

其中 $\mathbf{e}_u, \mathbf{e}_i$ 是可学习的embedding参数。

第 $\ell$ 层GAT更新为：

$$
\mathbf{h}_u^{(\ell)} = \text{GAT}\left(\mathbf{h}_u^{(\ell-1)}, \{\mathbf{h}_i^{(\ell-1)} \mid i \in \mathcal{N}_u^{\text{CF}}\}\right)
$$

$$
\mathbf{h}_i^{(\ell)} = \text{GAT}\left(\mathbf{h}_i^{(\ell-1)}, \{\mathbf{h}_u^{(\ell-1)} \mid u \in \mathcal{N}_i^{\text{CF}}\}\right)
$$

其中 $\mathcal{N}_u^{\text{CF}}$ 表示用户 $u$ 交互过的物品集合，$\mathcal{N}_i^{\text{CF}}$ 表示与物品 $i$ 交互过的用户集合。

**GAT聚合函数**（高层形式）：

$$
\text{GAT}(\mathbf{h}_v, \{\mathbf{h}_u\}_{u \in \mathcal{N}_v}) = \sum_{u \in \mathcal{N}_v} \alpha_{vu} \mathbf{W} \mathbf{h}_u
$$

其中 $\alpha_{vu}$ 是通过attention机制学习的邻居权重：

$$
\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]))}{\sum_{u' \in \mathcal{N}_v} \exp(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_{u'}]))}
$$

最终的CF view表示为：

$$
\mathbf{h}_u^{\text{CF}} = \mathbf{h}_u^{(L)}, \quad \mathbf{h}_i^{\text{CF}} = \mathbf{h}_i^{(L)}
$$

### 2.2 KG View (Knowledge Graph View)

#### GNN架构选择

**使用HeteroConv + GAT**，原因：
- 处理异构图：不同类型的边（long-term, short-term, describes）需要独立的参数
- 保留语义差异：长期兴趣和短期兴趣应该有不同的聚合模式
- GAT提供自适应聚合能力

#### 可学习Entity Mask

在KG view中，我们首先对entity embedding应用**可学习的mask**：

$$
\tilde{\mathbf{e}}_e = m_e \odot \mathbf{e}_e, \quad m_e = \sigma(\mathbf{z}_e)
$$

其中：
- $\mathbf{z}_e$ 是可学习的logit参数
- $\sigma(\cdot)$ 是sigmoid函数，确保 $m_e \in [0, 1]$
- $\odot$ 是element-wise乘法

**Mask初始化**：基于entity频率 $f_e$（在所有KG三元组中的出现次数）：

$$
m_e^{(0)} = \min\left(1, \frac{f_e}{f_{\text{median}}}\right)
$$

**设计动机**：
- **去噪**：LLM提取的entity质量不一，低质量entity应该被抑制
- **端到端学习**：Mask通过梯度下降学习，自动发现对推荐有用的entity
- **可解释性**：Mask值反映entity的重要性，可用于分析和可视化

#### 编码过程

设初始embedding为：

$$
\mathbf{h}_u^{(0)} = \mathbf{e}_u, \quad \mathbf{h}_e^{(0)} = \tilde{\mathbf{e}}_e, \quad \mathbf{h}_i^{(0)} = \mathbf{e}_i
$$

第 $\ell$ 层异构GNN更新为：

$$
\mathbf{h}_u^{(\ell)} = \text{AGG}_{\text{sum}}\left(
\begin{cases}
\text{GAT}_{\text{long}}\left(\mathbf{h}_u^{(\ell-1)}, \{\mathbf{h}_e^{(\ell-1)} \mid (u,e) \in \mathcal{E}_{\text{long}}\}\right) \\
\text{GAT}_{\text{short}}\left(\mathbf{h}_u^{(\ell-1)}, \{\mathbf{h}_e^{(\ell-1)} \mid (u,e) \in \mathcal{E}_{\text{short}}\}\right) \\
\text{GAT}_{\text{rev\_long}}\left(\mathbf{h}_u^{(\ell-1)}, \{\mathbf{h}_e^{(\ell-1)} \mid (e,u) \in \mathcal{E}_{\text{rev\_long}}\}\right) \\
\text{GAT}_{\text{rev\_short}}\left(\mathbf{h}_u^{(\ell-1)}, \{\mathbf{h}_e^{(\ell-1)} \mid (e,u) \in \mathcal{E}_{\text{rev\_short}}\}\right)
\end{cases}
\right)
$$

$$
\mathbf{h}_e^{(\ell)} = \text{AGG}_{\text{sum}}\left(
\begin{cases}
\text{GAT}_{\text{rev\_desc}}\left(\mathbf{h}_e^{(\ell-1)}, \{\mathbf{h}_i^{(\ell-1)} \mid (i,e) \in \mathcal{E}_{\text{rev\_desc}}\}\right) \\
\text{GAT}_{\text{long}}\left(\mathbf{h}_e^{(\ell-1)}, \{\mathbf{h}_u^{(\ell-1)} \mid (u,e) \in \mathcal{E}_{\text{long}}\}\right) \\
\text{GAT}_{\text{short}}\left(\mathbf{h}_e^{(\ell-1)}, \{\mathbf{h}_u^{(\ell-1)} \mid (u,e) \in \mathcal{E}_{\text{short}}\}\right)
\end{cases}
\right)
$$

$$
\mathbf{h}_i^{(\ell)} = \text{AGG}_{\text{sum}}\left(
\begin{cases}
\text{GAT}_{\text{desc}}\left(\mathbf{h}_i^{(\ell-1)}, \{\mathbf{h}_e^{(\ell-1)} \mid (e,i) \in \mathcal{E}_{\text{desc}}\}\right)
\end{cases}
\right)
$$

**关键设计**：
- 不同边类型使用**独立的GAT层**（$\text{GAT}_{\text{long}}, \text{GAT}_{\text{short}}, \text{GAT}_{\text{desc}}$ 等）
- 同一节点类型的多个边类型聚合使用**sum**（保留所有信息）

最终的KG view表示为：

$$
\mathbf{h}_u^{\text{KG}} = \mathbf{h}_u^{(L)}, \quad \mathbf{h}_i^{\text{KG}} = \mathbf{h}_i^{(L)}, \quad \mathbf{h}_e^{\text{KG}} = \mathbf{h}_e^{(L)}
$$

### 2.3 多视图对比学习 (Multi-View Contrastive Learning)

**设计动机**：
- CF view和KG view学到的表示应该**相似但互补**
- 相似性：同一用户在两个视图中应该接近（一致性）
- 互补性：两个视图提供不同的信息源（CF信号 vs. 知识信号）

我们使用**InfoNCE loss**在batch内对比：

$$
\mathcal{L}_{\text{contrast}} = -\frac{1}{|\mathcal{B}|} \sum_{u \in \mathcal{B}} \log \frac{\exp(\text{sim}(\bar{\mathbf{h}}_u^{\text{CF}}, \bar{\mathbf{h}}_u^{\text{KG}}) / \tau)}{\sum_{u' \in \mathcal{B}} \exp(\text{sim}(\bar{\mathbf{h}}_u^{\text{CF}}, \bar{\mathbf{h}}_{u'}^{\text{KG}}) / \tau)}
$$

其中：
- $\mathcal{B}$ 是当前batch的用户集合
- $\bar{\mathbf{h}} = \mathbf{h} / \|\mathbf{h}\|_2$ 是L2归一化后的embedding
- $\text{sim}(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\top \mathbf{b}$ 是余弦相似度
- $\tau$ 是温度超参数（控制分布的平滑度）

**正负样本定义**：
- 正样本：同一用户 $u$ 在CF view和KG view的表示 $(\mathbf{h}_u^{\text{CF}}, \mathbf{h}_u^{\text{KG}})$
- 负样本：batch内其他用户 $u'$ 的cross-view相似度 $(\mathbf{h}_u^{\text{CF}}, \mathbf{h}_{u'}^{\text{KG}})$

**为什么使用batch-wise对比**：
- 全局对比（所有用户）计算复杂度高 $O(|\mathcal{U}|^2)$
- Batch-wise对比提供足够的监督信号且高效 $O(|\mathcal{B}|^2)$

---

## 3. 多视图融合 (Multi-View Fusion)

将CF view和KG view的表示融合为最终的用户/物品表示：

$$
\mathbf{h}_u^{\text{fused}} = \text{MLP}_u([\mathbf{h}_u^{\text{CF}} \| \mathbf{h}_u^{\text{KG}}])
$$

$$
\mathbf{h}_i^{\text{fused}} = \text{MLP}_i([\mathbf{h}_i^{\text{CF}} \| \mathbf{h}_i^{\text{KG}}])
$$

其中：
- $[\cdot \| \cdot]$ 表示concatenation
- $\text{MLP}(\cdot)$ 是2层全连接网络，包含ReLU激活和Dropout

**设计动机**：
- **Concatenation**: 保留两个视图的完整信息（vs. 简单平均会丢失细节）
- **MLP**: 学习自适应的融合权重（不同用户可能更依赖CF或KG）
- **独立的MLP**: User和Item的融合模式可能不同

**其他设计选择**：
- 不使用Residual connection（实验中发现效果相当，简化设计）
- 不使用Layer normalization（Dropout已经提供足够的正则化）

---

## 4. 训练目标 (Training Objective)

### 4.1 总损失函数

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \alpha \mathcal{L}_{\text{contrast}} + \beta \mathcal{L}_{\text{align}} + \gamma \mathcal{L}_{\text{mask}}
$$

其中 $\alpha, \beta, \gamma$ 是权重超参数。

**权重系数的设计理念**：
- $\mathcal{L}_{\text{rec}}$ 权重为1（主任务）
- $\alpha, \beta, \gamma \ll 1$（辅助任务，防止过度影响主任务）
- $\alpha > \beta > \gamma$（对比学习 > 对齐损失 > Mask正则）

### 4.2 推荐损失 $\mathcal{L}_{\text{rec}}$

我们使用**InfoNCE**作为推荐损失（对比学习formulation）：

$$
\mathcal{L}_{\text{rec}} = -\frac{1}{|\mathcal{B}|} \sum_{(u,i^+) \in \mathcal{B}} \log \frac{\exp(\mathbf{h}_u^{\text{fused}} \cdot \mathbf{h}_{i^+}^{\text{fused}} / \tau_{\text{rec}})}{\exp(\mathbf{h}_u^{\text{fused}} \cdot \mathbf{h}_{i^+}^{\text{fused}} / \tau_{\text{rec}}) + \sum_{i^- \in \mathcal{N}^-} \exp(\mathbf{h}_u^{\text{fused}} \cdot \mathbf{h}_{i^-}^{\text{fused}} / \tau_{\text{rec}})}
$$

其中：
- $(u, i^+)$ 是正样本pair（用户 $u$ 交互过的物品 $i^+$）
- $\mathcal{N}^-$ 是负采样的物品集合
- $\tau_{\text{rec}}$ 是温度参数

**设计动机**：
- InfoNCE优于传统BPR loss（更强的梯度信号，收敛更快）
- 负采样策略：随机负采样（简单高效）

### 4.3 多视图对比损失 $\mathcal{L}_{\text{contrast}}$

见 Section 2.3。

**作用**：确保CF view和KG view学到的用户表示保持一致性，避免两个视图"各说各话"。

### 4.4 Entity-Item对齐损失 $\mathcal{L}_{\text{align}}$

**设计动机**：
- Entity和它描述的Item应该在embedding空间中**接近**
- 例如："noir风格"这个entity应该接近"具有noir海报的电影"
- 这种对齐增强了Entity作为语义桥梁的能力

形式化定义：

$$
\mathcal{L}_{\text{align}} = -\frac{1}{|\mathcal{E}_{\text{desc}}|} \sum_{(e,i^+) \in \mathcal{E}_{\text{desc}}} \log \frac{\exp(\mathbf{h}_e^{\text{KG}} \cdot \mathbf{h}_{i^+}^{\text{KG}})}{\exp(\mathbf{h}_e^{\text{KG}} \cdot \mathbf{h}_{i^+}^{\text{KG}}) + \sum_{i^- \in \mathcal{N}^-} \exp(\mathbf{h}_e^{\text{KG}} \cdot \mathbf{h}_{i^-}^{\text{KG}})}
$$

其中：
- $(e, i^+)$ 是entity-item边（entity $e$ 描述了item $i^+$）
- $\mathcal{N}^-$ 是负采样的item集合（随机采样）
- **注意**：使用KG view的embedding（$\mathbf{h}^{\text{KG}}$），而非fused embedding

**为什么使用KG view而非fused**：
- Alignment loss的目标是让KG view学到语义一致的表示
- Fused embedding已经混合了CF信号，会稀释这种语义对齐

### 4.5 Mask正则化 $\mathcal{L}_{\text{mask}}$

**设计动机**：
- **稀疏性**：大部分entity应该被保留（mask $\approx 1$），只过滤少数低质量entity
- **确定性**：避免模棱两可的mask值（应该接近0或1），使模型易于解释

形式化定义：

$$
\mathcal{L}_{\text{mask}} = \lambda_{\text{sparse}} \sum_{e \in \mathcal{E}} (1 - m_e) - \lambda_{\text{entropy}} \cdot H(m)
$$

其中：
- 第一项是**L1稀疏正则**：最小化 $\sum (1 - m_e)$ 等价于最大化 $\sum m_e$（鼓励保留）
- 第二项是**熵正则**（负号表示最小化熵）：

$$
H(m) = -\frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \left[ m_e \log m_e + (1 - m_e) \log(1 - m_e) \right]
$$

- 最小化熵 $\Rightarrow$ mask接近0或1（确定性）

**权重系数**：
- $\lambda_{\text{sparse}} > 0$：控制保留entity的倾向
- $\lambda_{\text{entropy}} > 0$：控制确定性的强度

---

## 5. 推理 (Inference)

给定用户 $u$ 和候选物品 $i$，推荐分数计算为：

$$
\hat{y}_{ui} = \mathbf{h}_u^{\text{fused}} \cdot \mathbf{h}_i^{\text{fused}}
$$

推荐Top-K物品：

$$
\text{Rank}(u) = \text{argsort}_{i \in \mathcal{I}} \{\hat{y}_{ui}\}_{\text{desc}}[:K]
$$

**推理时的特殊处理**：
- Entity mask已经学习完成，直接使用 $m_e$
- 不需要重新计算对比损失或对齐损失
- 只需forward pass得到 $\mathbf{h}_u^{\text{fused}}$ 和 $\mathbf{h}_i^{\text{fused}}$

---

## 6. 模型变体 (Model Variants for Ablation)

为了验证各模块的有效性，我们设计了以下消融实验变体：

| 变体名称 | CF View | KG View | Contrast | Align | Mask | 说明 |
|---------|---------|---------|----------|-------|------|------|
| **Ours-Full** | ✅ | ✅ | ✅ | ✅ | ✅ | 完整模型 |
| **Ours-CF-Only** | ✅ | ❌ | ❌ | ❌ | ❌ | 只用协同过滤 |
| **Ours-KG-Only** | ❌ | ✅ | ❌ | ✅ | ✅ | 只用知识图谱 |
| **Ours-wo-Contrast** | ✅ | ✅ | ❌ | ✅ | ✅ | 无多视图对比 |
| **Ours-wo-Mask** | ✅ | ✅ | ✅ | ✅ | ❌ | 无Entity mask |

**实现方式**：
- CF-Only: `use_kg_view=False`，融合层退化为identity
- KG-Only: `use_cf_view=False`，融合层退化为identity
- wo-Contrast: `use_contrast=False`，$\mathcal{L}_{\text{contrast}} = 0$
- wo-Mask: `use_mask=False`，$m_e \equiv 1$

---

## 7. 关键设计决策总结

### 7.1 为什么使用GAT而非GCN？

- **自适应聚合**：推荐场景中邻居重要性差异大（热门物品 vs. 长尾物品）
- **缓解过平滑**：Attention机制防止信息在多跳传播中被稀释
- **实验验证**：在我们的数据集上GAT优于GCN约1-2% NDCG@10

### 7.2 为什么使用两个独立视图而非单一异构图？

- **信号解耦**：CF信号（用户行为相似性）和KG信号（语义相似性）本质不同
- **可解释性**：可以分析每个视图的贡献，理解模型决策
- **对比学习**：两个视图才能做对比，单一视图无法应用对比损失
- **灵活性**：可以独立调整每个视图的参数（层数、attention heads等）

### 7.3 为什么区分长期/短期兴趣？

- **捕获兴趣演化**：用户品味有稳定部分（长期）和探索部分（短期）
- **提升推荐多样性**：长期兴趣保证relevance，短期兴趣鼓励exploration
- **不同的聚合模式**：两种边类型在GNN中使用独立参数，学习不同的消息传递模式

### 7.4 为什么使用可学习Mask而非固定阈值？

- **端到端优化**：Mask通过梯度下降学习，自适应数据分布
- **避免人工调参**：无需手动设定entity过滤阈值
- **可解释性**：Mask值反映entity对推荐任务的重要性，可用于知识质量分析

### 7.5 为什么使用InfoNCE而非BPR？

- **更强的梯度信号**：InfoNCE在分母中包含所有负样本，提供更多监督
- **理论优势**：InfoNCE是互信息的下界，理论上更优
- **实验验证**：在我们的数据集上InfoNCE收敛更快且效果更好

---

## 8. 与Baseline的关键区别

### vs. KGAT
- **知识来源**：KGAT使用metadata KG（genre, director等），我们使用LLM提取的视觉KG
- **知识质量**：我们的KG更细粒度（32K+ 知识点 vs. KGAT的数百个metadata）
- **Mask机制**：我们有可学习mask过滤低质量entity，KGAT没有
- **双视图**：我们显式解耦CF和KG视图并对比学习，KGAT是单一异构图

### vs. LightGCN
- **知识增强**：LightGCN只用User-Item交互图，我们额外使用KG
- **多视图对比**：我们有CF view vs. KG view对比，LightGCN没有
- **Entity对齐**：我们有Entity-Item对齐损失，LightGCN没有entity概念

### vs. MKGAT (最强baseline)
- **知识来源**：MKGAT使用预训练多模态特征（ViT等），我们使用LLM提取的符号化知识
- **可解释性**：我们的知识是符号化的（"noir style", "warm color"），MKGAT的特征是黑盒向量
- **知识建模**：我们通过异构图建模用户兴趣，MKGAT直接融合特征向量
- **Mask机制**：我们有端到端的知识过滤，MKGAT没有

---

## 附录：符号表 (Notation Table)

| 符号 | 含义 |
|------|------|
| $\mathcal{U}, \mathcal{I}, \mathcal{E}$ | 用户、物品、知识实体集合 |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | 异构图 |
| $\mathcal{E}_{\text{long}}, \mathcal{E}_{\text{short}}$ | 长期/短期兴趣边 |
| $\mathcal{E}_{\text{desc}}$ | Entity描述Item的边 |
| $\mathbf{e}_u, \mathbf{e}_i, \mathbf{e}_e$ | 用户、物品、entity的初始embedding |
| $\mathbf{h}_u^{(l)}$ | 用户在第$l$层的隐藏状态 |
| $\mathbf{h}_u^{\text{CF}}, \mathbf{h}_u^{\text{KG}}$ | 用户在CF/KG视图的最终表示 |
| $\mathbf{h}_u^{\text{fused}}$ | 融合后的用户表示 |
| $m_e$ | Entity $e$ 的mask值（$\in [0,1]$） |
| $\mathcal{L}_{\text{rec}}$ | 推荐损失 (InfoNCE) |
| $\mathcal{L}_{\text{contrast}}$ | 多视图对比损失 |
| $\mathcal{L}_{\text{align}}$ | Entity-Item对齐损失 |
| $\mathcal{L}_{\text{mask}}$ | Mask正则化损失 |
| $\alpha, \beta, \gamma$ | 损失权重超参数 |
| $\tau$ | 温度超参数 |

---

**文档版本**: v1.0
**最后更新**: 2026-01-19
**代码依据**:
- `src/model/ours.py`: 主模型
- `src/model/encoders.py`: CF/KG编码器
- `src/model/losses.py`: 损失函数
- `src/data/graph_builder.py`: 图构建
