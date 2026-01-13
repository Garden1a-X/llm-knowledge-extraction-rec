# 下周任务规划（2026-01-14至2026-01-19）

*Created: 2026-01-13*

---

## 🎯 本周vs下周

### 本周完成（2026-01-08至2026-01-13）✅
- ✅ Phase 5/6/7：用户兴趣、格式转换、模型实现
- ✅ Baseline对比：BPR, LightGCN, KGAT
- ✅ Ours-Full训练：NDCG@10 = 0.1549 (+27.1% vs BPR)
- ✅ 文档修复：9个过时文档全部更新

### 本周进行中（论文撰写）
- 📝 Method章节撰写
- 📝 相关工作调研
- 📝 初步实验结果整理

### 下周重点（实验+论文）
1. **多模态图谱baseline**（MMGCN, MGAT）- 最重要
2. **LLM/MLLM推荐baseline**（学弟负责）
3. **特殊消融实验**（KG质量对比）
4. **消融实验**（我们模型的组件）
5. **论文继续撰写**（Experiments章节）

---

## 📋 任务分配

### 👨‍🎓 学弟任务：LLM/MLLM推荐baseline

#### 任务1：LLM推荐方法（2-3个）

**推荐模型**：
- [ ] **LLM4Rec** - LLM作为推荐器
- [ ] **TALLRec** - LLM增强推荐
- [ ] **RecLLM** - 推荐系统+LLM（可选）

**要求**：
- 使用ML-1M数据集（5-core filtered）
- 数据分割：70/10/20，Time-based ordering
- 评估指标：NDCG@10, Recall@10, Precision@10, Hit@10
- 评估模式：Full ranking（不是uni50）
- 至少运行1次（有时间可以3-5次取均值）

**交付**：
- 每个模型的NDCG@10结果
- 训练日志和配置文件
- 简要实现说明（用了哪个开源实现，如何适配数据）

**资源**：
- 代码仓库链接：（待查找开源实现）
- 数据位置：`/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m/`
- 可用GPU：询问后分配

**时间**：3-4天

---

#### 任务2：MLLM推荐方法（2个）

**推荐模型**：
- [ ] **VIP5** - 视觉指令微调推荐
- [ ] **LlamaRec** 或 **InstructRec** - Llama微调推荐

**要求**：同上（与LLM推荐一致）

**注意事项**：
- MLLM方法可能需要视觉特征，检查是否需要提取海报特征
- 如果需要海报，位置：`/data/xuao/llm-knowledge-extraction-rec/data/raw/ml-1m/posters/`

**时间**：3-4天

---

#### 任务清单模板（给学弟）

```markdown
## LLM/MLLM推荐Baseline实验

### 实验配置
- 数据集：MovieLens-1M (5-core filtered)
- 数据分割：70% train / 10% val / 20% test
- 排序：Time-based ordering
- 评估模式：Full ranking
- 指标：NDCG@10 (主), Recall@10, Precision@10, Hit@10

### 实验记录

#### 1. LLM4Rec
- [ ] 查找开源实现
- [ ] 适配ML-1M数据
- [ ] 运行训练
- [ ] 记录结果：NDCG@10 = ?

#### 2. TALLRec
- [ ] 查找开源实现
- [ ] 适配ML-1M数据
- [ ] 运行训练
- [ ] 记录结果：NDCG@10 = ?

#### 3. VIP5
- [ ] 查找开源实现
- [ ] 检查是否需要视觉特征
- [ ] 适配ML-1M数据
- [ ] 运行训练
- [ ] 记录结果：NDCG@10 = ?

#### 4. LlamaRec
- [ ] 查找开源实现
- [ ] 适配ML-1M数据
- [ ] 运行训练
- [ ] 记录结果：NDCG@10 = ?

### 遇到的问题
- 记录遇到的问题和解决方案

### 参考资料
- 论文PDF链接
- 开源代码仓库链接
```

---

### 🔬 我的任务：多模态图谱baseline + 消融实验

#### 任务1：多模态图谱方法 ⭐⭐⭐（最重要）

**为什么重要**：这些方法与我们的工作最相关，是核心对比baseline

**模型**：
- [ ] **MMGCN** - Multi-modal Graph Convolution Network (ACM MM 2019)
- [ ] **MGAT** - Multimodal Graph Attention Network (IPM 2020)
- [ ] **LATTICE** - Latent Type Information with KG (SIGIR 2021)（可选）

**实现方案**：
1. 查找官方代码：
   - MMGCN: https://github.com/weiyinwei/MMGCN
   - MGAT: https://github.com/zltao/MGAT
2. 如果官方代码适配困难，考虑：
   - 找第三方实现
   - 或简化实现（只用核心思想）
   - 或跳过LATTICE，只跑MMGCN+MGAT

**多模态特征**：
- 视觉：需要从海报提取CNN/ViT特征
- 文本：使用电影标题/类型（ML-1M自带）
- （音频：ML-1M没有，跳过）

**时间估算**：
- 环境搭建 + 特征提取：1天
- MMGCN实现与运行：1-2天
- MGAT实现与运行：1-2天
- **总计**：3-5天

---

#### 任务2：特殊消融实验 ⭐⭐⭐（证明LLM-KG质量）

**实验目的**：证明我们LLM提取的KG比传统外部KG更有效

**实验设计**：

**对照组A：KGAT（传统KG推荐）**
- KGAT + 外部KG（Freebase/DBpedia）
- vs KGAT + 我们的LLM-KG（已有：0.1209）

**对照组B：MMGCN（多模态图谱）**（如果时间允许）
- MMGCN + 标准多模态特征（CNN visual features）
- vs MMGCN + 我们的LLM知识点（替换visual features）

**数据准备**：
- [ ] 找到KGAT在ML-1M上使用的外部KG数据
  - 可能来源：KGAT官方仓库、KG4RecEval
  - 或使用Freebase/DBpedia的movie子集
- [ ] 格式转换为RecBole格式

**预期结果**：
- KGAT + 外部KG：预计 ~0.12-0.13（与我们的0.1209类似或略好）
- **论文叙事**：即使KGAT用外部KG略好，我们的方法（0.1549）仍然大幅领先
- **关键贡献**：不是简单的"KG好"，而是"LLM-KG + User兴趣 + Mask机制"的组合

**时间估算**：1-2天

---

#### 任务3：消融实验（我们模型的组件）

**配置已准备好**，只需运行：
- [ ] ours_wo_contrast.yaml - 去掉多视图对比学习
- [ ] ours_wo_mask.yaml - 去掉可学习Mask机制
- [ ] ours_kg_only.yaml - 只用KG视图
- [ ] ours_cf_only.yaml - 只用CF视图

**运行方式**：
```bash
# 并行运行（GPU 0,5,6,7）
CUDA_VISIBLE_DEVICES=0 python scripts/train_model_fast.py --config configs/ours_wo_contrast.yaml &
CUDA_VISIBLE_DEVICES=5 python scripts/train_model_fast.py --config configs/ours_wo_mask.yaml &
CUDA_VISIBLE_DEVICES=6 python scripts/train_model_fast.py --config configs/ours_kg_only.yaml &
CUDA_VISIBLE_DEVICES=7 python scripts/train_model_fast.py --config configs/ours_cf_only.yaml &
```

**预期结果**：
- w/o Contrast：预计下降3-5%（验证对比学习有效）
- w/o Mask：预计下降2-4%（验证Mask对抗幻觉）
- KG-only：预计下降10-15%（CF视图很重要）
- CF-only：预计下降5-10%（KG视图有贡献）

**时间估算**：0.5天（运行） + 0.5天（分析）

---

## 📅 下周时间表

### Day 1-2（周一/周二）：多模态特征准备 + MMGCN
- [ ] 提取视觉特征（CNN/ViT）
- [ ] 准备文本特征
- [ ] 搭建MMGCN环境
- [ ] 运行MMGCN

### Day 3-4（周三/周四）：MGAT + 特殊消融
- [ ] 运行MGAT
- [ ] 查找KGAT外部KG数据
- [ ] 运行KGAT + 外部KG

### Day 5（周五）：消融实验 + 结果整理
- [ ] 运行4个消融配置
- [ ] 整理所有实验结果
- [ ] 更新EXPERIMENT_TRACKING.md

### Day 6-7（周末）：结果分析 + 论文
- [ ] 创建对比表格和可视化
- [ ] 撰写Experiments章节
- [ ] 根据结果决定是否改进模型

---

## 🎯 关键决策点

### 决策1：MMGCN/MGAT实现难度
- **如果很难**：简化实现或跳过MGAT，只跑MMGCN
- **如果容易**：两个都跑

### 决策2：外部KG获取
- **如果容易找到**：做特殊消融实验（证明LLM-KG优势）
- **如果很难找到**：论文中引用KGAT原论文的结果（间接对比）

### 决策3：根据LLM/MLLM结果
- **如果LLM/MLLM很强**（>0.15）：考虑改进我们的模型（加入LLM特征）
- **如果LLM/MLLM一般**（<0.14）：保持现有模型，强调我们方法的有效性

### 决策4：消融实验结果
- **如果某些组件贡献小**：论文中诚实报告，讨论原因
- **如果所有组件都重要**：强调方法的精心设计

---

## 💡 论文写作要点

### Method章节（本周）
- 3.1 双阶段知识点提取
- 3.2 动态用户兴趣提取
- 3.3 知识增强异构图推荐模型

### Experiments章节（下周）
- 4.1 实验设置（数据集、评估协议、实现细节）
- 4.2 Baseline对比
  - 传统方法（BPR）
  - 图谱方法（LightGCN, KGAT）
  - 多模态图谱（MMGCN, MGAT）
  - LLM/MLLM推荐
  - **我们的方法显著领先** ⭐
- 4.3 消融实验
  - 模型组件有效性
  - **LLM-KG vs 外部KG**（如果做了）
- 4.4 结果分析
  - 案例研究
  - Mask权重分析
  - 可视化

---

## 📊 预期最终结果表

| 类别 | 方法 | NDCG@10 | vs Ours |
|------|------|---------|---------|
| **传统CF** | BPR | 0.1219 | -27.1% |
| **图谱** | LightGCN | 0.1267 | -22.3% |
| **图谱** | KGAT | 0.1209 | -28.1% |
| **多模态图谱** | MMGCN | ~0.13-0.14? | ~-10-20%? |
| **多模态图谱** | MGAT | ~0.13-0.14? | ~-10-20%? |
| **LLM推荐** | LLM4Rec | 待运行 | ? |
| **LLM推荐** | TALLRec | 待运行 | ? |
| **MLLM推荐** | VIP5 | 待运行 | ? |
| **MLLM推荐** | LlamaRec | 待运行 | ? |
| **Ours** | **Ours-Full** | **0.1549** | **baseline** ⭐ |
| **消融** | w/o Contrast | ~0.147? | -5% |
| **消融** | w/o Mask | ~0.151? | -2% |
| **消融** | KG-only | ~0.135? | -13% |
| **消融** | CF-only | ~0.145? | -6% |

---

## ⚠️ 风险与应对

### 风险1：MMGCN/MGAT实现困难
- **应对**：简化实现，或只跑一个，或引用原论文结果

### 风险2：LLM/MLLM结果太好
- **应对**：改进我们的模型，或强调我们方法的可解释性/效率优势

### 风险3：消融实验结果不理想
- **应对**：诚实报告，讨论原因，可能需要调整模型

### 风险4：外部KG数据难获取
- **应对**：论文中引用KGAT原论文结果作为间接对比

---

## 📝 更新日志

### 2026-01-13
- 创建下周任务规划
- 明确学弟任务（LLM/MLLM推荐）
- 规划多模态图谱baseline实验
- 设计特殊消融实验（LLM-KG质量）

---

*Created: 2026-01-13*
*For week: 2026-01-14 to 2026-01-19*
