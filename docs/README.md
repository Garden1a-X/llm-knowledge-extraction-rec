# 文档说明

本目录包含项目的所有规划和设计文档。

## 📁 当前文档（2026-01-11更新）

### **核心文档**

1. **KNOWLEDGE_EXTRACTION_PLAN.md** ⭐ 总体规划
   - LLM知识抽取完整方案
   - 双阶段提取流程（7个Phase）
   - 分层聚类标准化
   - 用户兴趣提取
   - 图谱构建与Mask机制
   - **对应论文Method章节的三个部分**
   - 最后更新：2026-01-04

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

4. **EXPERIMENT_TRACKING.md**
   - 实验进度实时跟踪
   - 已完成实验的结果记录
   - 结果汇总表
   - 代码仓库链接

### **工作日志**

1. **WORK_LOG_20260111_PHASE3.md** ⭐ 最新
   - **Phase 3实现完成！** 🎉 提取脚本ready
   - 完整Prompt设计（165 entities全部列出）
   - 提取脚本实现（增量+错误恢复+实时统计）
   - Token优化：2200 tokens（比预估低24%）
   - 成本更低：$0.26 for 630部（比预估$0.34低）
   - **准备运行Phase 3验证（~800部电影）**

2. **WORK_LOG_20260111.md**
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

### **已完成**：
- ✅ Phase 1: 小规模探索（170部电影，1869个知识点）
- ✅ Phase 2a: Relation聚类+LLM微调（128 → 16个标准relations）
- ✅ **Phase 2b: Entity聚类+LLM Refinement（626 → 165个标准entities）** 🎉
  - Stage 1: Filtering（725 → 479 kept）
  - Stage 1.5: Redistribution（626 unique，零流失）
  - Stage 2: BERTopic聚类（626 → 133 clusters）
  - Stage 3: LLM三步refinement（Split→Merge→Rename）
  - 标准vocabulary生成（165 entities + 13 噪声）

### **正在进行**：
- 🔄 **Phase 3: 验证与扩充**（实现完成，待运行）✅
  - ✅ Prompt设计完成（docs/PHASE3_PROMPT_DESIGN.md）
  - ✅ 提取脚本完成（scripts/phase3_validate_vocabulary.py）
  - ⏸️ 待运行：20%数据验证（约800部电影）
  - 使用165个标准entities作为vocabulary v1
  - 限制提取 + 覆盖率统计
  - 如果覆盖率<90%，扩充vocabulary v1 → v2

### **下一步**：
- Phase 3: 验证与扩充（20%数据，约800部电影）
- Phase 4: 全量提取（80%数据，约3100部电影）
- Phase 5: 用户兴趣提取
- Phase 6: 知识图谱构建（User-Knowledge-Item异构图）
- Phase 7: 推荐模型训练（带Mask机制的GNN）

**进度**：Phase 2完成 ≈ **40%整体进度** 🎉

详见 `KNOWLEDGE_EXTRACTION_PLAN.md` 末尾的"下一步行动"章节。

---

*Last updated: 2026-01-11*
