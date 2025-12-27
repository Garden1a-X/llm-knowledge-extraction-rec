# 文档说明

本目录包含项目的所有规划和设计文档。

## 📁 当前文档（2024-12-27更新）

### **核心文档**

1. **KNOWLEDGE_EXTRACTION_PLAN.md** ⭐ 最新
   - LLM知识抽取完整方案
   - 双阶段提取流程（7个Phase）
   - 分层聚类标准化
   - 用户兴趣提取
   - 图谱构建与Mask机制
   - **对应论文Method章节的三个部分**

2. **BASELINE_PLAN.md**
   - Baseline实验完整规划
   - 13个对比方法的分类和选择理由
   - 3条演进路径
   - 消融实验设计

3. **EXPERIMENT_TRACKING.md**
   - 实验进度实时跟踪
   - 已完成实验的结果记录
   - 结果汇总表
   - 代码仓库链接

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

## 🚀 下一步

参考 `KNOWLEDGE_EXTRACTION_PLAN.md` 开始实现：
- Phase 1: 小规模探索（5%数据，~200部电影）
- Phase 2: 双层聚类（关系≤15类，实体≤30/类）
- Phase 3-7: 验证→全量→用户兴趣→图谱→模型

---

*Last updated: 2024-12-27*
