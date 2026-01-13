# 文档说明

本目录包含项目的所有规划和设计文档。

## 📁 当前文档（2026-01-13更新）

### **核心文档**

1. **KNOWLEDGE_EXTRACTION_PLAN.md** ⭐ 总体规划
   - LLM知识抽取完整方案
   - 双阶段提取流程（7个Phase）
   - 分层聚类标准化
   - 用户兴趣提取
   - 图谱构建与Mask机制
   - **对应论文Method章节的三个部分**
   - **状态**: Phase 1-7全部完成 ✅
   - 最后更新：2026-01-11

2. **RELATION_DEFINITIONS.md** ⭐ Relation边界定义（新增）
   - 14+1个标准Relations的完整定义
   - 每个relation的职责、范围、边界规则
   - 复杂边界cases的判断标准（visual_theme等）
   - 为LLM prompt设计和人工审核提供参考
   - 创建时间：2026-01-04

3. **BASELINE_PLAN.md**
   - Baseline实验完整规划
   - 13个对比方法的分类和选择理由
   - 3条演进路径
   - 消融实验设计

4. **EXPERIMENT_TRACKING.md** ⭐ 实验跟踪
   - 实验进度实时跟踪
   - Baseline对比结果（BPR, LightGCN, KGAT）
   - Ours-Full结果：NDCG@10 = 0.1549
   - 消融实验计划
   - **状态**: 持续更新中
   - 最后更新：2026-01-13

5. **MODEL_DESIGN.md** ⭐ 模型架构设计
   - 异构图结构（User-Entity-Item）
   - 双视图编码器（CF + KG）
   - 损失函数设计（4种损失）
   - 可学习Mask机制
   - **状态**: 已实现并完成训练 ✅

6. **METHOD_IMPROVEMENTS.md** 方法改进方向
   - 负采样策略改进
   - 动态温度调整
   - 双层对比学习
   - Hard negative sampling
   - **状态**: 计划中，未实施

7. **KG4RECEVAL_GUIDE.md** KGAT Baseline指南
   - KG4RecEval使用方法
   - KGAT训练步骤
   - 数据准备与配置
   - **结果**: 最终用RecBole直接跑KGAT（0.1209）

8. **PHASE5_USER_INTERESTS.md** 用户兴趣提取
   - Hybrid统计+LLM方法
   - 短期/长期兴趣提取
   - RecBole格式输出
   - **状态**: 已完成 ✅

### **工作日志**

1. **WORK_LOG_20260111_PHASE3_ITERATION.md** ⭐ 最新
   - **Phase 3 Prompt优化与Self-Review实现** 🎉
   - 两轮Prompt迭代（强制NEW_ → Self-Review）
   - NEW_使用率提升10倍（1.67% → 16.2%）
   - 有效覆盖率94.9%（超过90%目标）✅
   - Self-review机制证明有效
   - **正在运行完整683部验证**

2. **WORK_LOG_20260111_PHASE3.md**
   - Phase 3实现完成 - 提取脚本ready
   - 完整Prompt设计（165 entities全部列出）
   - 提取脚本实现（增量+错误恢复+实时统计）
   - Token优化与成本分析

3. **WORK_LOG_20260111.md**
   - **Phase 2b完成！** 🎉 Entity聚类与LLM Refinement
   - Stage 2: BERTopic聚类（626 → 133 clusters）
   - Stage 3: LLM三步refinement（Split → Merge → Rename）
   - 关键创新：CoT Reasoning + Business Context
   - 成功修复：superhero → hero（不再是villain）
   - 标准Entity词汇表提取（165个标准entities）

2. **WORK_LOG_20260108.md**
   - Phase 2b Stage 1 & 1.5完成
   - Filtering（725 → 479 kept + 246 removed）
   - Redistribution（626 unique entities，零流失）
   - Stage 2方法重新设计（Pure LLM失败 → Embedding+LLM）

3. **WORK_LOG_20260104.md**
   - Phase 2b Entity重分配方案设计
   - Relation边界混淆问题分析
   - 两阶段entity重分配策略
   - 关键决策记录（depicted_subject vs character_type等）

4. **WORK_LOG_20250102.md** ⚠️ 已过时
   - Phase 2a初步工作记录（LLM微调之前）
   - 记录14个relations的初步聚类结果
   - 仅作历史参考，实际最终结果是16个relations

---

## 🗂️ 归档文档

`archive/` 目录下的文件已过时，仅供参考：
- `PROJECT_CHECK_20241226.txt` - 2024-12-26的项目检查（已被EXPERIMENT_TRACKING.md替代）
- `development_plan.py` - Python格式的开发计划（已被KNOWLEDGE_EXTRACTION_PLAN.md替代）
- `experiment_design.py` - Python格式的实验设计（已被BASELINE_PLAN.md替代）

---

## 📖 阅读顺序建议

**如果你是新加入的成员**：
1. 先看 `BASELINE_PLAN.md` - 了解整体实验框架
2. 再看 `KNOWLEDGE_EXTRACTION_PLAN.md` - 了解核心方法
3. 最后看 `EXPERIMENT_TRACKING.md` - 了解当前进度

**如果你要实现代码**：
1. 参考 `KNOWLEDGE_EXTRACTION_PLAN.md` 的完整实现流程
2. 按照 Phase 1-7 的顺序实现

**如果你要写论文**：
1. Method部分参考 `KNOWLEDGE_EXTRACTION_PLAN.md` 的章节结构
2. Experiments部分参考 `BASELINE_PLAN.md` 和 `EXPERIMENT_TRACKING.md`

---

## 🎯 论文对应关系

### **Method章节**（参考 KNOWLEDGE_EXTRACTION_PLAN.md）
- 3.1 双阶段知识点提取（Two-Stage Knowledge Extraction）
- 3.2 动态用户兴趣点提取（Dynamic User Interest Extraction）
- 3.3 基于多模态知识图谱的推荐（MMKG-based Recommendation）

### **Experiments章节**（参考 BASELINE_PLAN.md + EXPERIMENT_TRACKING.md）
- 4.1 实验设置
- 4.2 Baseline对比（13个方法）
- 4.3 消融实验（图谱有效性 + LLM能力影响）
- 4.4 结果分析

---

## 🚀 当前状态与下一步

### **已完成** ✅：
- ✅ **Phase 1-7: 知识提取与模型实现全部完成** 🎉
  - Phase 1: 小规模探索（170部电影，1869个知识点）
  - Phase 2a: Relation聚类+LLM微调（128 → 16个标准relations）
  - Phase 2b: Entity聚类+LLM Refinement（626 → 165个标准entities）
  - Phase 3: 词汇验证（20%数据，覆盖率85.6%）
  - Phase 4: 全量电影知识提取（3,415电影，32,675 KPs）
  - Phase 5: 用户兴趣提取（6,040用户，~90K edges）
  - Phase 6: RecBole格式转换
  - Phase 7: 模型实现与训练

- ✅ **Baseline对比实验**
  - BPR: NDCG@10 = 0.1219 ± 0.0021
  - LightGCN: NDCG@10 = 0.1267 ± 0.0013 (+3.9% vs BPR)
  - KGAT: NDCG@10 = 0.1209 (-0.8% vs BPR) ⚠️
  - **Ours-Full: NDCG@10 = 0.1549 (+27.1% vs BPR)** ⭐⭐⭐

- ✅ **关键发现**:
  - 我们的方法显著超越所有baseline（+22-28%）
  - KGAT验证了传统KG推荐的局限性（比BPR还低）
  - 可引用KG4RecEval论文：KGAT的KGER=-0.026

### **进行中** 🔄：
- 📝 **文档修复**: 更新所有过时/错误的文档
- ⏸️ **消融实验**: 4个配置已准备，待运行
- 🏃 **tune1**: 可能仍在运行，待查看

### **下一步** 📋：
1. **消融实验**（高优先级）- 验证各组件有效性
2. **结果分析与整理** - 创建对比表格和可视化
3. **方法改进**（可选）- 负采样、温度调整等
4. **论文撰写** - Method + Experiments章节

**进度**：核心实验完成 ≈ **85%整体进度** 🎉

详见 `KNOWLEDGE_EXTRACTION_PLAN.md` 末尾的"下一步行动"章节。

---

*Last updated: 2026-01-13*
