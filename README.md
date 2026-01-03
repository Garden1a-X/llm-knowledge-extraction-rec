# 基于LLM知识提取的多模态推荐系统

> **多模态知识图谱推荐 - MovieLens 1M**

## 🎯 项目简介

本项目利用**多模态大语言模型（MLLM）**从**电影海报**中提取细粒度视觉知识点，构建知识增强的异构图谱，用于电影推荐。

### **核心创新点**

1. **双阶段知识点提取**
   - 小规模自由探索 → 数据驱动发现知识类型
   - 分层聚类标准化（关系层 + 实体层）
   - 验证扩充 → 全量限制提取

2. **动态用户兴趣提取**
   - 基于历史观影的知识偏好聚合
   - TF-IDF加权强调用户独特性

3. **知识增强异构图谱**
   - User-Knowledge-Item 三层异构图
   - 知识点作为桥梁连接用户和电影
   - Mask机制抑制低质量知识点

4. **可解释性**
   - 基于知识点的推荐解释
   - 可视化用户视觉偏好

---

## 📊 数据集

### MovieLens 1M + 电影海报

- **基础数据集**：[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
  - 6,040 用户
  - 3,900 电影
  - 1,000,209 评分
  - 5-core 过滤后：~3,700 电影，~900K 评分

- **海报数据**：
  - 来源：[@11Li11](https://github.com/11Li11/Li/tree/master/ml-1m)
  - 3,882 张电影海报（JPG格式）

**数据结构**：
```
data/raw/ml-1m/
├── ratings.dat          # 评分记录
├── users.dat            # 用户信息
├── movies.dat           # 电影信息
└── posters/             # 海报图片
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

---

## 🏗️ 技术架构

### **Phase 1: LLM知识提取**（离线）

```
电影海报
  ↓ MLLM (GPT-4o-mini / Qwen3-VL-8B)
  ↓ 提取<关系, 实体>对
视觉知识点
  ↓ BGE Embedding + HDBSCAN聚类
  ↓ 双层标准化（关系层 + 实体层）
标准化知识词典
  ↓ 最多15类关系，每类最多30种实体
知识图谱构建
```

**10种知识点关系类型**（数据驱动发现）：
- Color_Palette（色彩风格）
- Visual_Mood（视觉氛围）
- Composition_Style（构图风格）
- Lighting_Effect（光影效果）
- Era_Aesthetic（时代风格）
- Genre_Visual（类型视觉特征）
- Character_Display（角色呈现）
- Setting_Environment（场景类型）
- Poster_Design（海报设计）
- Visual_Symbolism（视觉符号）

### **Phase 2: 知识图谱推荐**（在线）

```
User-Knowledge-Item 异构图
  ↓ 图神经网络 (GAT/GCN)
  ↓ Mask机制（抑制低质量知识点）
  ↓ 多模态融合
用户/物品表示
  ↓ 预测评分
推荐结果 + 解释
```

---

## 📁 项目结构

```
llm-knowledge-extraction-rec/
├── data/                          # 数据目录
│   ├── raw/ml-1m/                 # 原始MovieLens数据
│   ├── recbole/                   # RecBole格式数据
│   ├── processed/                 # 知识提取结果
│   └── graphs/                    # 知识图谱
│
├── src/                           # 源代码
│   ├── data/                      # 数据加载
│   ├── extraction/                # LLM知识提取
│   ├── clustering/                # 知识点聚类
│   ├── graph/                     # 图谱构建
│   ├── model/                     # 推荐模型
│   └── utils/                     # 工具函数
│
├── baselines/                     # Baseline方法
│   ├── prepare_data_for_recbole.py  # 数据准备（5-core过滤）
│   ├── run_baseline.py              # 训练单个baseline
│   └── run_multiple_trials.py       # 多次试验
│
├── scripts/                       # 实验脚本
│   ├── 01_extract_knowledge_pilot.py   # Phase 1: 5%探索
│   ├── 02_cluster_knowledge.py         # Phase 2: 聚类
│   ├── 03_validate_vocabulary.py       # Phase 3: 验证
│   ├── 04_extract_knowledge_full.py    # Phase 4: 全量
│   ├── 05_extract_user_interests.py    # Phase 5: 用户兴趣
│   └── 06_build_graph.py               # Phase 6: 图谱构建
│
├── docs/                          # 📚 详细文档
│   ├── KNOWLEDGE_EXTRACTION_PLAN.md   # 知识提取完整方案 ⭐
│   ├── BASELINE_PLAN.md               # Baseline实验规划
│   ├── EXPERIMENT_TRACKING.md         # 实验进度跟踪
│   └── README.md                      # 文档导航
│
├── configs/                       # 配置文件
├── tests/                         # 测试代码
├── requirements.txt               # Python依赖
├── README.md                      # 本文件
└── SETUP_GUIDE.md                 # 环境配置指南
```

---

## 📚 详细文档

**完整的实现方案和实验设计请查看 `docs/` 目录：**

### **核心文档**

1. **[KNOWLEDGE_EXTRACTION_PLAN.md](docs/KNOWLEDGE_EXTRACTION_PLAN.md)** ⭐ 最重要
   - 完整的7个Phase实现流程
   - 论文Method章节的3个部分
   - 双阶段提取 + 分层聚类 + Mask机制
   - 成本估算：~$1.23

2. **[BASELINE_PLAN.md](docs/BASELINE_PLAN.md)**
   - 13个对比方法的完整规划
   - 3条演进路径
   - 消融实验设计

3. **[EXPERIMENT_TRACKING.md](docs/EXPERIMENT_TRACKING.md)**
   - 实时实验进度跟踪
   - 结果汇总表
   - 代码仓库链接

4. **[docs/README.md](docs/README.md)**
   - 文档导航和阅读顺序

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone <repository-url>
cd llm-knowledge-extraction-rec

# 创建conda环境
conda create -n llm_kg_rec python=3.9
conda activate llm_kg_rec

# 安装依赖
pip install -r requirements.txt
```

详见 [SETUP_GUIDE.md](SETUP_GUIDE.md)

### 2. 数据准备

```bash
# 下载MovieLens 1M数据到 data/raw/ml-1m/

# 准备RecBole格式数据（带5-core过滤）
python baselines/prepare_data_for_recbole.py \
    --ml_data_dir data/raw/ml-1m \
    --output_dir data/recbole/ml-1m \
    --min_interactions 5
```

### 3. 运行Baseline方法

```bash
# BPR (传统协同过滤)
python baselines/run_multiple_trials.py \
    --model BPR \
    --num_trials 5 \
    --epochs 300 \
    --device cuda

# LightGCN (图谱方法)
python baselines/run_multiple_trials.py \
    --model LightGCN \
    --num_trials 5 \
    --epochs 300 \
    --device cuda
```

输出文件：`outputs/baselines/multiple_trials/MODEL_ml_1m_summary_5trials.json`

### 4. LLM知识提取（核心工作）

参考 [KNOWLEDGE_EXTRACTION_PLAN.md](docs/KNOWLEDGE_EXTRACTION_PLAN.md) 的7个Phase：

```bash
# Phase 1: 小规模探索（5%数据）
python scripts/01_extract_knowledge_pilot.py

# Phase 2: 双层聚类
python scripts/02_cluster_knowledge.py

# Phase 3: 验证与扩充（20%数据）
python scripts/03_validate_vocabulary.py

# Phase 4: 全量提取（80%数据）
python scripts/04_extract_knowledge_full.py

# Phase 5: 用户兴趣提取
python scripts/05_extract_user_interests.py

# Phase 6: 知识图谱构建
python scripts/06_build_graph.py
```

---

## 📊 当前进展

### ✅ 已完成（Phase 1）

- ✅ 项目架构设计
- ✅ 5-core 数据过滤
- ✅ BPR baseline：NDCG@10 = 0.1219 ± 0.0021
- ✅ LightGCN baseline：NDCG@10 = 0.1267 ± 0.0013 (+3.9%)
- ✅ 知识提取完整方案设计
- ✅ 实验框架搭建

### 🚧 进行中

- 实现知识提取模块（Phase 1-7）

### 📋 待完成

**知识提取与图谱构建**：
- [ ] Phase 1: 小规模探索（5%数据）
- [ ] Phase 2: 双层聚类
- [ ] Phase 3-4: 验证与全量提取
- [ ] Phase 5: 用户兴趣提取
- [ ] Phase 6: 知识图谱构建
- [ ] Phase 7: 推荐模型实现

**Baseline方法**：
- [ ] KGAT（知识图谱方法）
- [ ] MMGCN, MGAT（多模态图谱方法）⭐ 最相关
- [ ] VBPR（多模态方法）
- [ ] LLM4Rec, TALLRec（LLM推荐）
- [ ] VIP5, LlamaRec（MLLM推荐）

**实验与分析**：
- [ ] 消融实验（图谱有效性 + LLM能力影响）
- [ ] 性能对比与统计检验
- [ ] 案例分析与可视化

---

## 🛠️ 技术栈

- **Baseline框架**：RecBole
- **MLLM**：GPT-4o-mini（主） / Qwen3-VL-8B（消融）
- **Embedding**：BGE (BAAI/bge-base-en-v1.5)
- **聚类**：HDBSCAN / K-means
- **图神经网络**：PyTorch Geometric
- **深度学习**：PyTorch

---

## 📈 实验结果（持续更新）

| 方法 | NDCG@10 | Recall@10 | 提升 vs BPR |
|------|---------|-----------|-------------|
| **传统方法** |
| BPR | 0.1219 ± 0.0021 | 0.0598 ± 0.0004 | - |
| **图谱方法** |
| LightGCN | 0.1267 ± 0.0013 | 0.0581 ± 0.0009 | +3.9% |
| **多模态图谱** |
| MMGCN | - | - | - |
| MGAT | - | - | - |
| **Ours方法** |
| Ours-Full | - | - | - |

详见 [EXPERIMENT_TRACKING.md](docs/EXPERIMENT_TRACKING.md)

---

## 📖 参考文献

**多模态推荐**：
- MMGCN: Wei et al. Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. ACM MM 2019.
- MGAT: Tao et al. Multimodal Graph Attention Network for Recommendation. IPM 2020.
- VBPR: He & McAuley. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. AAAI 2016.

**图谱推荐**：
- LightGCN: He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.
- KGAT: Wang et al. KGAT: Knowledge Graph Attention Network for Recommendation. KDD 2019.
- NGCF: Wang et al. Neural Graph Collaborative Filtering. SIGIR 2019.

**LLM推荐**：
- 待补充

---

## 📝 更新日志

### 2026-01-03
- ✅ **完成Phase 2a LLM微调**（Relation映射完成）
- ✅ 使用GPT-4o-mini增量式微调，修正embedding聚类的语义问题
- ✅ 最终**16个标准relation**：artistic_style, character_type, color_palette, composition_style, costume_design, depicted_subject, dominant_color, genre, graphic_element, interaction, lighting, mood, others_relation, symbolism, text_style, visual_effect
- ✅ 覆盖率100%，质量评分90/100
- ✅ Bug修复：修复split操作中orphan检测失败问题

### 2026-01-02
- ✅ 完成Phase 2a Relation聚类初步工作（128 → 14标准relation）
- ✅ 实现双聚类方法对比（BERTopic vs Agglomerative）
- ✅ 实现finalize_mapping + 手动优化机制
- ✅ 发现问题：embedding被`_type`后缀主导，需要LLM微调

### 2025-12-27
- ✅ 完成Phase 1 Baseline实验（BPR, LightGCN）
- ✅ 创建完整的知识提取方案（KNOWLEDGE_EXTRACTION_PLAN.md）
- ✅ 文档结构优化（归档旧文档）

### 2024-12-26
- ✅ 集成RecBole框架
- ✅ 5-core数据过滤
- ✅ 实验跟踪系统建立

### 2024-07-25
- 探索知识点标准化方案
- BGE + K-means聚类实验

### 2024-07-17
- Demo全流程实验（200+电影）
- 10类知识点类型设计

### 2024-07-01
- 项目启动
- 方法构想和技术方案设计

---

## 📧 联系方式

如有问题或建议，欢迎提Issue或PR。

## 📄 许可证

MIT License

---

*Last updated: 2026-01-03*
