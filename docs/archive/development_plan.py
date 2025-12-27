#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发计划 - LLM视觉知识提取推荐系统

项目日期: 2024-12-25
版本: MovieLens 1M 电影推荐实验

本文档详细说明了整个项目的开发计划，包括baseline实现和创新方法实验。

================================================================================
项目概览
================================================================================

使用LLM从电影海报提取结构化视觉知识点，构建知识图谱用于电影推荐。

核心创新:
- 使用LLM（GPT-4o mini / Qwen3-VL）提取视觉知识而非CNN特征
- 构建三层知识图谱（电影-知识点、用户-兴趣、用户-电影）
- 与多个baseline对比验证效果

================================================================================
方案A：Baseline基线模型
================================================================================

目的：建立对比基准，证明LLM方法的优势


方法1: LightGCN - 纯协同过滤baseline（耗时 3-5天）
-----------------------------------------------------
描述: 不使用任何知识图谱，仅用用户-物品交互

论文: LightGCN: Simplifying and Powering Graph Convolution Network
      for Recommendation (SIGIR 2020)

实现任务:
- [ ] 构建user-item交互图
- [ ] 实现LightGCN模型
- [ ] 训练评估
- [ ] 保存结果

文件:
- src/model/lightgcn.py
- scripts/train_lightgcn.py

技术栈: PyTorch, PyTorch Geometric


方法2: KGAT - 传统知识图谱推荐（耗时 5-7天）
-----------------------------------------------------
描述: 使用传统KG（元数据）的推荐系统

论文: KGAT: Knowledge Graph Attention Network for Recommendation (KDD 2019)

知识图谱结构:
```
User <交互> Movie <有属性> Entity
                 |
                 +---> Genre/Director/Year等
```

实现任务:
- [ ] 构建MovieLens元数据图谱（5种关系）
  - Movie -> Genre
  - Movie -> Year
  - Movie -> Title entities
- [ ] 实现KGAT attention机制
- [ ] 实现knowledge-aware embedding
- [ ] 训练评估

文件:
- src/graph/kg_builder.py - 构建知识图谱
- src/model/kgat.py - KGAT模型
- scripts/train_kgat.py

知识图谱来源: MovieLens自带genres等元数据


方法3: RippleNet - 传播机制（耗时 5-7天）
-----------------------------------------------------
描述: 使用偏好传播机制

论文: RippleNet: Propagating User Preferences on the Knowledge Graph (CIKM 2018)

核心思想: 在用户历史交互的KG上进行ripple传播

实现任务:
- [ ] 使用KGAT构建的图谱
- [ ] 实现ripple传播机制
- [ ] 多跳传播聚合
- [ ] 训练评估

文件:
- src/model/ripplenet.py
- scripts/train_ripplenet.py


方法4: MKGAT - 多模态知识图谱（最重要对比，耗时 7-10天）
-----------------------------------------------------
描述: 整合视觉特征到baseline中（最核心对比）

论文: Multi-modal Knowledge Graphs for Recommender Systems (参考相关工作)

核心区别:
- MKGAT: 使用CNN提取的视觉特征（ResNet等）
- 我们的方法: 使用LLM提取的结构化视觉知识

实现任务:
- [ ] 使用预训练CNN提取视觉特征
  - ResNet50或ViT
  - 对3882个5张海报提取特征
- [ ] 构建多模态知识图谱
  - 文本关系（genres等）
  - 视觉关系（视觉embedding）
- [ ] 实现MKGAT模型
- [ ] 训练评估

文件:
- src/extraction/visual_features.py - CNN特征提取
- src/model/mkgat.py
- scripts/train_mkgat.py


================================================================================
方案B：创新方法实验
================================================================================

阶段1：LLM基础搭建（耗时 2-3天）
-----------------------------------------------------

任务5. 配置LLM客户端
- [ ] 集成OpenAI API
- [ ] 测试GPT-4o mini
- [ ] 设计提示词模板
- [ ] Token使用监控

文件: src/extraction/llm_client.py


阶段2：知识提取试点（耗时 3-5天）
-----------------------------------------------------

任务6. 5张海报知识提取 - 试点（100部电影）

知识点类别（10种）:
1. Color_Palette - 色彩风格
2. Visual_Mood - 视觉氛围
3. Composition - 构图风格
4. Lighting - 光线处理
5. Texture - 质感风格
6. Character_Presentation - 角色呈现
7. Era_Style - 时代风格
8. Genre_Visual - 类型视觉标签
9. Typography - 文字排版
10. Others - 其他视觉特征

实现任务:
- [ ] 设计知识提取prompt
- [ ] 实现5张海报提取函数
- [ ] 对100部电影进行试点
- [ ] 评估结果质量可行性
- [ ] 调整优化策略

文件:
- src/extraction/movie_extractor.py
- scripts/02_extract_knowledge_pilot.py
- data/processed/knowledge_pilot_100.json

API成本估算（100部）:
- 约 $0.05 (试点成本很低)


阶段3：知识标准化（耗时 3-5天）
-----------------------------------------------------

任务7. 知识点聚类标准化

目的: LLM输出的知识表述各异（如"warm tone","暖色调","orange palette"）

聚类方法:
- [ ] 使用BGE模型生成knowledge point embeddings
- [ ] 对相似知识点进行聚类
- [ ] HDBSCAN或K-means聚类
- [ ] 生成标准化知识点词典
- [ ] 保存知识点分类体系

文件:
- src/clustering/embedder.py
- src/clustering/clusterer.py
- scripts/03_cluster_knowledge.py
- data/processed/knowledge_vocab.json


阶段4：全量知识提取（耗时 2-3天）（取决于API速度/并发）
-----------------------------------------------------

任务8. 5张海报知识提取 - 全量（3900部电影）

实现任务:
- [ ] 批量处理所有电影
- [ ] 异常处理和重试
- [ ] 进度保存和断点续传
- [ ] 使用标准化知识点词典

文件:
- scripts/02_extract_knowledge_full.py
- data/processed/movie_knowledge_full.json

API成本估算（全量）: 约$1.43（见下方详细计算）


阶段5：用户兴趣提取（耗时 3-5天）
-----------------------------------------------------

任务9. 用户兴趣提取

核心思想: 基于用户历史评分的5张海报，推断用户对视觉知识点的偏好

实现任务:
- [ ] 构建用户兴趣提取逻辑
- [ ] 聚合用户正向反馈
- [ ] 计算兴趣/负面兴趣
- [ ] 生成用户兴趣知识图谱

文件:
- src/extraction/user_extractor.py
- scripts/04_extract_user_interests.py
- data/processed/user_interests.json


阶段6：知识图谱构建（耗时 3-5天）
-----------------------------------------------------

任务10. 构建三层知识图谱

图谱结构:
```
1. 电影-知识点关系
   Movie --has--> Knowledge_Point

2. 用户-兴趣关系
   User --interested_in--> Knowledge_Point

3. 用户-电影交互关系
   User --rated(score)-> Movie
   (正向: rating >= 4, 负向: rating <= 2)
```

实现任务:
- [ ] 构建三层图结构
- [ ] 保存图数据
- [ ] 测试NetworkX或PyG格式
- [ ] 可视化验证

文件:
- src/graph/builder.py
- src/graph/storage.py
- scripts/05_build_graph.py


阶段7：推荐模型训练（耗时 7-10天）
-----------------------------------------------------

任务11. Relation-aware GNN推荐模型

创新架构:
- 多关系图神经网络（知识 + 交互）
- Relation-aware attention
- 对比学习
- 负采样策略

实现任务:
- [ ] 实现多关系GNN架构
- [ ] Relation-aware attention层
- [ ] 对比学习损失
- [ ] 训练pipeline
- [ ] 评估指标

文件:
- src/model/gnn.py
- src/model/trainer.py
- src/model/evaluator.py
- scripts/06_train_model.py


阶段8：实验对比（耗时 3-5天）
-----------------------------------------------------

任务12. 全面实验对比

实验设置:
- 数据集: MovieLens 1M
- 划分: 70% train, 10% val, 20% test
- 评估指标: NDCG@10, Recall@10, Precision@10, Hit@10

对比方法:
1. LightGCN (no KG)
2. KGAT (metadata KG)
3. RippleNet (metadata KG)
4. MKGAT (visual features)
5. **Ours** (LLM visual knowledge)

分析内容:
- [ ] 全面性能对比
- [ ] 统计显著性检验
- [ ] Case study（案例分析）
- [ ] 知识点分析

文件:
- scripts/07_run_experiments.py
- scripts/08_analyze_results.py
- outputs/results/comparison.csv


================================================================================
API成本估算（GPT-4o mini）
================================================================================

GPT-4o mini 定价（2024年12月）
-------------------------------
- Input: $0.150 / 1M tokens
- Output: $0.600 / 1M tokens

5张海报知识提取成本
-------------------

单部电影成本:
- 输入: 视觉理解（~170 tokens）+ prompt（~300 tokens）= ~470 tokens
- 输出: 知识点JSON（~500 tokens）
- 单部总计: ~970 tokens

100部试点:
- 总tokens: 100 × 970 = 97,000 tokens ≈ 0.1M tokens
- 成本: $0.015 + $0.018 ≈ **$0.03**

3900部全量:
- 总tokens: 3,900 × 970 = 3,783,000 tokens ≈ 3.8M tokens
- Input成本: 3.8M × 0.5 × $0.150/1M = $0.285
- Output成本: 3.8M × 0.5 × $0.600/1M = $1.14
- 成本: **$1.43**

用户兴趣提取成本（6,040个用户）:
- 每用户: ~500 tokens（输入聚合+输出兴趣）
- 总tokens: 6,040 × 500 = 3,020,000 tokens ≈ 3M tokens
- 成本: **$0.90**

总计（加上重试余量）:
- 预计成本: $1.43 + $0.90 = $2.33
- 容错余量: 约2倍
- **总预算: $5-10**


节省策略
--------

1. 分阶段执行
   - 先100部试点（$0.03）
   - 验证500部效果（$0.15）
   - 最后全量3900部（$1.43）

2. 质量验证
   - 提前检验LLM输出质量
   - 优化prompt减少重试

3. 异步批处理
   - 使用批量API（如有折扣）
   - 多进程加速但控制并发


================================================================================
时间规划
================================================================================

快速方案（仅核心baseline） - 4-6周
-----------------------------------

```
Week 1-2: Baseline实现
  - LightGCN (3天)
  - KGAT (4天)
  - 初步基准

Week 3-4: 创新方法核心
  - LLM客户端 (2天)
  - 知识提取试点100部 (3天)
  - 知识标准化 (4天)

Week 5-6: 完整实验
  - 全量提取 (2天)
  - 图谱构建 (3天)
  - 模型训练 (4天)
  - 实验对比 (3天)
```

完整方案（所有baseline） - 8-10周
----------------------------------

```
Week 1-4: 所有Baseline
  - LightGCN (3天)
  - KGAT (5天)
  - RippleNet (5天)
  - MKGAT (7天)
  - 完整基准 (4天)

Week 5-8: 创新方法（同上）

Week 9-10: 深入分析
  - 消融实验
  - 案例分析
  - 论文撰写
```


================================================================================
当前进度
================================================================================

第0阶段 - 环境搭建
------------------
- [x] 项目初始化
- [x] 环境配置（PyTorch + PyG）
- [x] 数据集准备
- [x] MovieLens 1M数据验证（3883电影, 6040用户, 1M评分, 3882海报）

接下来任务
----------
- 待定

当前 milestone
--------------
1. 最快路径: LightGCN实现（建立初步baseline）
2. 异步准备: LLM客户端集成
3. 第一个milestone: LightGCN训练完成 + 100部电影知识提取完成


================================================================================
项目结构
================================================================================

```
llm-knowledge-extraction-rec/
├── docs/
│   ├── development_plan.py          # 本文件
│   └── experiment_design.py         # 实验设计
├── src/
│   ├── model/
│   │   ├── lightgcn.py          # LightGCN baseline
│   │   ├── kgat.py              # KGAT baseline
│   │   ├── ripplenet.py         # RippleNet baseline
│   │   ├── mkgat.py             # MKGAT baseline
│   │   ├── gnn.py               # 我们的模型
│   │   ├── trainer.py           # 训练函数
│   │   └── evaluator.py         # 评估函数
│   ├── extraction/
│   │   ├── llm_client.py        # LLM客户端
│   │   ├── movie_extractor.py   # 5张海报知识提取
│   │   ├── user_extractor.py    # 用户兴趣提取
│   │   └── visual_features.py   # CNN特征（MKGAT使用）
│   ├── graph/
│   │   ├── kg_builder.py        # KG构建（baseline使用）
│   │   ├── builder.py           # 三层图谱（我们的模型）
│   │   └── storage.py           # 存储管理
│   └── clustering/
│       ├── embedder.py          # Embedding生成
│       └── clusterer.py         # 聚类算法
├── scripts/
│   ├── 01_test_data.py                  # 第0阶段
│   ├── 02_extract_knowledge_pilot.py    # 试点提取
│   ├── 02_extract_knowledge_full.py     # 全量提取
│   ├── 03_cluster_knowledge.py          # 知识聚类
│   ├── 04_extract_user_interests.py     # 用户兴趣
│   ├── 05_build_graph.py                # 图谱构建
│   ├── 06_train_lightgcn.py             # LightGCN
│   ├── 06_train_kgat.py                 # KGAT
│   ├── 06_train_ripplenet.py            # RippleNet
│   ├── 06_train_mkgat.py                # MKGAT
│   ├── 06_train_ours.py                 # 我们的模型
│   ├── 07_run_experiments.py            # 批量实验
│   └── 08_analyze_results.py            # 结果分析
└── data/
    ├── processed/
    │   ├── knowledge_pilot_100.json    # 试点结果
    │   ├── movie_knowledge_full.json   # 全量结果
    │   ├── knowledge_vocab.json        # 标准化词典
    │   ├── user_interests.json         # 用户兴趣
    │   └── visual_features/            # CNN特征
    └── graphs/
        ├── kg_metadata.pkl             # 元数据KG
        ├── kg_visual.pkl               # 视觉KG
        └── kg_ours.pkl                 # 我们的KG
```


================================================================================
关键决策理由
================================================================================

为什么需要多个baseline？
------------------------

1. **LightGCN**: 证明KG的必要性（不使用KG）
2. **KGAT**: 使用传统KG的注意力机制
3. **RippleNet**: 使用偏好传播机制
4. **MKGAT**: 使用视觉特征但非LLM（最核心对比）

为什么选GPT-4o mini？
---------------------

- 成本合理：比GPT-4便宜约10倍
- 性能足够：对于视觉理解和知识提取，mini已经相当强
- 速度快：响应速度快，适合批量处理
- 可复现：相比本地模型，API更稳定一致
- 未来可替换：如果效果不好可以直接升级到GPT-4o

分阶段执行的好处
----------------

1. **降低风险**: 先验证LLM效果再全量提取
2. **快速迭代**: 试点阶段可以快速调整策略
3. **多重baseline**: 有充分对比才能体现创新价值
4. **成果可发**: 即使中途停止也有基线结果


================================================================================
项目摘要
================================================================================

- 项目启动日期: 2024-12-25
- LLM API: OpenAI GPT-4o mini
- 预算: $5-10
- 预计时间: 6-10周
- 核心milestone: 先完成第一个baseline和LLM试点

================================================================================

最后更新: 2024-12-25
下次milestone: 完成第一个baseline或LLM客户端
"""


# 方便导入的常量定义
KNOWLEDGE_CATEGORIES = [
    "Color_Palette",
    "Visual_Mood",
    "Composition",
    "Lighting",
    "Texture",
    "Character_Presentation",
    "Era_Style",
    "Genre_Visual",
    "Typography",
    "Others"
]

BASELINE_METHODS = [
    "LightGCN",
    "KGAT",
    "RippleNet",
    "MKGAT"
]

EVALUATION_METRICS = [
    "NDCG@10",
    "Recall@10",
    "Precision@10",
    "Hit@10"
]

API_COST_ESTIMATES = {
    "pilot_100": 0.03,
    "full_3900": 1.43,
    "user_interests": 0.90,
    "total_budget": (5, 10)
}


if __name__ == "__main__":
    print(__doc__)
