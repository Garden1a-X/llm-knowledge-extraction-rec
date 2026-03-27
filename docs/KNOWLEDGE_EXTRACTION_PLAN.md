# LLM知识抽取方案

*Created: 2025-12-27*
*Last Updated: 2026-01-13*
*Status: **Phase 1-7全部完成！** ✅ - 知识提取、模型实现、训练全部完成，Baseline对比结果优秀（NDCG@10=0.1549, +27.1% vs BPR）*

---

## 📝 工作日志

### 2026-01-11: Phase 2b Stage 2 & 3 完成！🎉 Entity聚类与LLM Refinement ✅

**重大突破**：Phase 2b完成，双层知识标准化全部完成！

- ✅ **Stage 2: BERTopic聚类完成**（626 → 133 canonical clusters）
  - 方法：BERTopic自适应聚类（UMAP + HDBSCAN）
  - 每个relation独立聚类，自动确定最佳cluster数
  - 输入：626 unique entities（671 instances）
  - 输出：133 canonical clusters
  - 压缩率：79.2%

- ✅ **Stage 3: LLM三步Refinement完成**（133 → 178 clusters）
  - **关键创新**: Split → Merge → Rename（正确顺序）
  - **CoT Reasoning**: analysis-before-decision JSON结构
  - **Business Context**: 添加推荐场景说明
  - **成功修复**: superhero → "hero"（不再是villain）✅
  - Round 1 + Round 2 refinement
  - 最终：178 clusters（165标准 + 13噪声）

- ✅ **标准Entity词汇表提取**
  - 工具：`scripts/extract_entity_vocabulary.py`
  - 输出：`results/standard_entity_vocabulary.json`
  - **165个标准entities** 跨15个relations
  - 13个噪声entities（other_*）
  - 5/15 relations达标（10-15范围）：33.3%
  - **准备进入Phase 3验证与扩充**！

- 📊 **关键技术突破**：
  1. **正确的顺序**：Split → Merge → Rename（避免Name Anchoring）
  2. **CoT的力量**：利用Transformer自回归生成，先分析再决策
  3. **Business Context**：明确推荐场景，NEVER merge opposites
  4. **全面错误处理**：字段验证、JSON截断优雅降级、缺失信息推断

- 🎯 **质量指标**：
  - 语义质量：✅ 无明显对立混合（hero ≠ villain分离成功）
  - 覆盖率：5/15达标，10/15可接受
  - 图谱密度：显著改善（165 vs 626）

详见 `docs/WORK_LOG_20260111.md`

### 2026-01-08: Phase 2b Stage 1 & 1.5完成，Stage 2方法重新设计 ✅
- ✅ **Stage 1 Filtering完成**: 所有15个relations处理完成
  - 输入：725个entities（含重复）
  - 输出：479 kept + 246 removed
  - Unique entities：626个
  - 质量：约95%准确率，边界区分清晰
- ✅ **Stage 1.5 Redistribution完成**: 零entity流失
  - 192个entities成功重分配到建议的relation
  - 54个entities已存在于目标relation（跳过）
  - 最终：626 → 626 unique entities，零流失 ✅
  - 总计数：725 → 671（减少54个冗余分布）
- ⚠️ **Stage 2 Pure LLM方法失败**
  - 尝试1：visual_theme 58→50（只减少13.8%）+ validation错误
  - 尝试2（改进prompt）：依然效果差，丢失6个entities
  - 根本问题：任务过于复杂，LLM倾向保守，难以达到目标压缩率
- ✅ **Stage 2方法重新设计**: 改用Embedding + LLM混合方法
  - 借鉴Phase 2a成功经验（128 relations → 16 relations）
  - 新流程：BGE Embedding → Agglomerative/HDBSCAN聚类 → LLM验证每个cluster
  - 优势：稳定性高、可控压缩率、LLM任务简化
  - 实现：`scripts/phase2b_stage2_entity_clustering.py`
  - 关键改进：动态确定n_clusters（不同relation的entity数差异大）
- 🔄 **下一步**: 测试新的Stage 2脚本，处理所有15个relations

### 2026-01-03: Phase 2a LLM微调完成 ✅
- ✅ **LLM微调执行完成**: 使用GPT-4o-mini修正embedding聚类的语义问题
  - 实现增量式微调（split/merge/separate/rename操作）
  - 两阶段LLM调用：主要修正 + orphan分配
  - 最终结果：**16个标准relation**（从14增加到16）
  - 保存: `relation_mapping_final_v2.json`
- ✅ **Bug修复**: 修复split操作中source名称复用导致的orphan检测失败
  - 问题：当split创建与source同名的新relation时，orphan未被检测
  - 解决：使用临时标记`__ORPHAN_FROM_<source>__`确保所有orphan被检测
  - 结果：全部11个orphan成功被第二次LLM调用处理
- ✅ **质量评估**:
  - 映射完整性：128/128个原始relation全部映射
  - 语义一致性：90/100分（仅3处小语义问题，可接受）
  - 覆盖率：100%
- ✅ **最终16个标准Relations**:
  1. action_behaviors (行为动作)
  2. additional_elements (附加元素，原others_relation)
  3. artistic_styles (艺术风格，复数形式)
  4. character_type (角色类型)
  5. color_palette (色彩搭配)
  6. composition_styles (构图风格)
  7. depicted_entities (描绘实体)
  8. depicted_subject (描绘主体)
  9. design_element (设计元素)
  10. genre (类型)
  11. lighting (光照)
  12. mood (情绪氛围)
  13. symbolism (象征意义)
  14. text_style (文字排版)
  15. texture (纹理质感)
  16. visual_theme (视觉主题，合并了dominant_color和theme)
- 🔄 **下一步**: Phase 2b Entity聚类（626 → 200-300个标准entity）

### 2026-01-02: Phase 2a Relation聚类与LLM微调实现 ✅
- ✅ **Phase 1提取完成**: 170个电影，新prompt成功
  - 128个唯一relation，1869个知识点
  - 626个唯一entity，平均复用2.99x
  - Prompt依赖度19.5%，质量优秀
- ✅ **Phase 2a规划**: 双层聚类方案确定
  - Embedding: BGE (BAAI/bge-base-en-v1.5)
  - 方法1: BERTopic（探索+可视化）
  - 方法2: Agglomerative（精确控制）
- ✅ **Relation聚类完成**:
  - 实现RelationClusterer模块
  - 对比BERTopic和Agglomerative结果
  - 最终方案: Agglomerative N=20 + 自动合并
  - 输出: **14个标准relation**
  - 保存: `results/relation_mapping_final.json`
- ⚠️ **发现问题**: Embedding聚类被`_type`等后缀主导
  - `action_type`混入了animal_type, setting_type, vehicle_type等
  - `genre`混入了narrative_*系列
  - 语义不一致，需要修正
- ✅ **LLM微调方案**: 使用GPT-4o-mini修正语义问题
  - 实现RelationRefiner模块 (`src/clustering/relation_refiner.py`)
  - 创建微调脚本 (`scripts/refine_relation_mapping.py`)
  - 利用LLM的语义理解能力，修正embedding聚类的后缀主导问题
- ✅ **LLM微调完成**: 2026-01-03
  - 执行成功，生成`relation_mapping_final_v2.json`
  - 结果：16个标准relation，128个原始relation全部映射
  - 质量：90/100分

### 2025-12-29: Phase 1 质量优化
- ✅ **首次提取完成**: 170个电影，全部成功
- ⚠️ **质量问题发现**:
  - Relation依赖度100%（只使用prompt中的8个例子）
  - Entity多样性80%（图谱极度稀疏，平均复用2.14次）
- ✅ **Prompt重新设计**:
  - 删除具体relation列表，鼓励多样性
  - 强调entity抽象化和可复用性
- ✅ **测试验证** (10个电影):
  - Relation: 8 → 43种 (+35)
  - Prompt依赖: 100% → 17.7% (-82.3%)
  - Entity复用: 1.28x → 1.53x
- ✅ **重新提取完成**: 用新prompt提取170个电影

### 2025-12-27: Phase 1 设计与实现
- 实现提取基础设施（增量保存、ID映射、实时恢复）
- 创建分析工具（notebook）

---

## 📋 概述

本文档详细描述了使用多模态大语言模型（MLLM）从电影海报中提取细粒度视觉知识点的完整流程。

### **核心创新**
1. **双阶段知识点提取**：小规模探索 → 验证扩充 → 全量提取
2. **分层知识标准化**：关系层聚类 + 实体层聚类
3. **动态用户兴趣提取**：基于历史观影的知识偏好
4. **Mask增强学习**：减少对低质量知识点的依赖

---

## 🎯 Method章节结构（论文）

### **3.1 双阶段知识点提取（Two-Stage Knowledge Extraction）**
- Phase 1: 小规模自由探索
- Phase 2: 双层聚类与标准化
- Phase 3: 验证与扩充
- Phase 4: 全量限制提取

### **3.2 动态用户兴趣点提取（Dynamic User Interest Extraction）**
- 基于历史观影的知识聚合
- 加权策略（评分、时间衰减）
- 用户兴趣表示

### **3.3 基于多模态知识图谱的推荐（MMKG-based Recommendation）**
- 三层异构图构建：User-Knowledge-Item
- 图神经网络架构
- Mask机制：抑制低质量知识点
- 推荐预测

---

## 🔄 完整流程

```
Phase 1: 小规模探索（5%数据）
  ↓ MLLM自由提取<关系, 实体>对

Phase 2: 双层聚类
  ↓ 关系层聚类（最多15类）
  ↓ 实体层聚类（每类最多30种）
  ↓ 构建标准词典 v1

Phase 3: 验证与扩充（20%数据，含5%）
  ↓ 限制提取（用v1词典）
  ↓ 统计覆盖率
  ↓ 如果覆盖率 < 90%，扩充词典 v1 → v2

Phase 4: 全量提取（80%数据）
  ↓ 限制提取（用v2词典）

Phase 5: 用户兴趣提取
  ↓ 基于历史观影聚合知识点

Phase 6: 知识图谱构建
  ↓ User-Knowledge-Item 三层异构图

Phase 7: 推荐模型训练
  ↓ 带Mask机制的GNN
```

---

## 📊 Phase 1: 小规模探索（5%数据）

### **目标**
从数据中**自动发现**知识点的关系类型和实体类型，而非预定义。

### **数据规模**
- **电影数量**：~200部（5% of ~3900部有海报的电影）
- **选择策略**：
  - 随机抽样，保证类型多样性
  - 优先选择评分数多的（代表性强）
  - 跨越不同年代和类型

### **MLLM模型**
- **GPT-4o-mini**（主要）
- **Qwen3-VL-8B**（消融实验用）

### **Prompt设计**

```
你是一个专业的电影海报视觉分析专家。请从这张电影海报中提取视觉知识点。

电影信息：
- 标题：{title}
- 类型：{genres}
- 年份：{year}

任务：提取海报的视觉特征，以<关系类型, 具体实体>的格式输出。

说明：
- 关系类型：视觉特征的类别（如：颜色风格、氛围、构图等）
- 具体实体：该类别下的具体描述词（如：暖色调、神秘感、对称构图等）
- 每部电影最多提取10个<关系, 实体>对
- 关系和实体都用简短的英文词组（2-4个单词）

输出格式（JSON）：
{
  "knowledge_points": [
    {"relation": "Color_Style", "entity": "warm_tones"},
    {"relation": "Visual_Mood", "entity": "mysterious_atmosphere"},
    {"relation": "Composition", "entity": "symmetrical_layout"},
    ...
  ]
}

要求：
1. 关系类型要能概括一类视觉特征（不要过于具体）
2. 实体要具体且有区分度（能区分不同电影）
3. 关系和实体都用snake_case命名
4. 最多10个知识点，选择最显著的特征
```

### **输出示例**

```json
{
  "movie_id": 1,
  "title": "Toy Story (1995)",
  "knowledge_points": [
    {"relation": "Color_Palette", "entity": "bright_primary_colors"},
    {"relation": "Color_Palette", "entity": "high_saturation"},
    {"relation": "Visual_Mood", "entity": "joyful_atmosphere"},
    {"relation": "Visual_Mood", "entity": "playful_energy"},
    {"relation": "Composition_Style", "entity": "character_centered"},
    {"relation": "Lighting_Effect", "entity": "bright_even_lighting"},
    {"relation": "Era_Aesthetic", "entity": "90s_cgi_style"},
    {"relation": "Genre_Visual", "entity": "animation_aesthetic"},
    {"relation": "Character_Display", "entity": "group_portrait"},
    {"relation": "Setting_Environment", "entity": "toy_world_theme"}
  ]
}
```

### **实现**

```python
# scripts/01_extract_knowledge_pilot.py

import openai
from pathlib import Path
import json
from tqdm import tqdm
import random

def select_pilot_movies(all_movies, ratio=0.05):
    """选择试点电影（5%）"""
    # 策略：类型多样性 + 评分数
    n_pilot = int(len(all_movies) * ratio)

    # 按类型分层抽样
    sampled = stratified_sample_by_genre(all_movies, n_pilot)

    return sampled

def extract_knowledge_with_gpt4o_mini(poster_path, movie_info):
    """用GPT-4o-mini提取知识点"""

    prompt = build_prompt(movie_info)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional movie poster visual analyst."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image(poster_path)}}
            ]}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return result['knowledge_points']

def main():
    # 1. 选择试点电影
    pilot_movies = select_pilot_movies(all_movies, ratio=0.05)
    print(f"Selected {len(pilot_movies)} pilot movies")

    # 2. 提取知识点
    results = []
    for movie in tqdm(pilot_movies):
        try:
            knowledge = extract_knowledge_with_gpt4o_mini(
                poster_path=f"data/raw/ml-1m/posters/{movie.id}.jpg",
                movie_info=movie
            )
            results.append({
                "movie_id": movie.id,
                "title": movie.title,
                "knowledge_points": knowledge
            })
        except Exception as e:
            print(f"Error on movie {movie.id}: {e}")
            continue

    # 3. 保存结果
    save_json(results, "data/processed/knowledge_pilot_raw.json")
    print(f"✓ Extracted knowledge from {len(results)} movies")
```

### **预期输出**

- **文件**：`data/processed/knowledge_pilot_raw.json`
- **内容**：200部电影的原始知识点（未聚类）
- **统计**：
  - 总知识点数：~2000个<关系, 实体>对
  - 唯一关系数：~50-100个
  - 唯一实体数：~500-1000个

### **成本估算**

- 200张图 × $0.0003/图 = **$0.06**

---

## 🔬 Phase 2: 双层聚类与标准化

### **目标**
将自由提取的知识点标准化为：
- **M+1个关系类**（最多15个 + 1个others）
- 每个关系下**K+1种实体**（最多30个 + 1个others）

### **2.1 关系层聚类**（✅ 已完成 2026-01-02）

#### **输入**
从Phase 1收集所有唯一的"关系"名称（如：`Color_Palette`, `Color_Style`, `Visual_Mood`, `Mood_Atmosphere`...）

**实际数据**：128个唯一relation，1869个实例

#### **方法选择与对比**

我们实现并对比了两种聚类方法：

**方法1: BERTopic（自动探索）**
- 优点：自动发现cluster数量，可视化工具丰富
- 缺点：对短关键词效果不佳，容易产生"junk bucket" clusters
- 实际结果：16 clusters，但Topic 0/1包含16个语义混杂的低频relations

**方法2: Agglomerative Clustering（精确控制）✅ 最终选择**
- 优点：精确控制cluster数量，适合短关键词
- 参数：N=20, Average linkage, Cosine distance
- 实际结果：20 clusters → 合并低频后得到15个标准relation

#### **最终实现**

```python
# src/clustering/relation_clusterer.py

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

def cluster_relations(raw_relations, n_clusters=20):
    """
    聚类关系类型

    Args:
        raw_relations: List of relation names
        n_clusters: 目标cluster数量

    Returns:
        relation_mapping: {原始关系 -> 标准关系}
        standard_relations: List of 标准关系名
    """

    # 1. 统计频率
    relation_freq = Counter(raw_relations)

    # 2. BGE embedding
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    embeddings = model.encode(list(relation_freq.keys()))

    # 3. Agglomerative聚类
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average',
        metric='cosine'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # 4. Finalize映射：合并低频clusters到others
    relation_mapping = finalize_relation_mapping(
        labels=cluster_labels,
        relation_freq=relation_freq,
        min_total_instances=10  # 总实例数<10的cluster合并到others
    )

    # 提取标准关系名
    standard_relations = sorted(set(relation_mapping.values()))

    return relation_mapping, standard_relations

def finalize_relation_mapping(labels, relation_freq, min_total_instances=10):
    """
    最终化映射：自动合并低频clusters到others_relation

    对每个cluster:
    - 如果总实例数 >= threshold: 保留，选最高频relation作为标准名
    - 如果总实例数 < threshold: 合并到others_relation
    """
    relation_mapping = {}

    for cluster_id in set(labels):
        cluster_relations = [rel for rel, lbl in zip(...) if lbl == cluster_id]
        total_instances = sum(relation_freq[rel] for rel in cluster_relations)

        if total_instances >= min_total_instances:
            # 选最高频的作为标准名
            standard_name = max(cluster_relations, key=lambda r: relation_freq[r])
            for rel in cluster_relations:
                relation_mapping[rel] = standard_name
        else:
            # 低频cluster → others
            for rel in cluster_relations:
                relation_mapping[rel] = 'others_relation'

    return relation_mapping
```

**初步结果**（2026-01-02）：
- 输入：128个唯一relation，1869个实例
- Agglomerative N=20 → 20 clusters
- 合并7个低频clusters（总实例数<10）→ others_relation
- 手动调整命名（3处）：提升语义清晰度
- 输出：**14个标准relation**（13个有效 + 1个others）
- 覆盖率：99.5%+
- ⚠️ 问题：部分cluster语义不一致（被后缀主导）

**LLM微调后最终结果**（2026-01-03）：
- 输入：14个初步relation + 128个原始relation
- LLM增量微调：split/merge/separate/rename操作
- 两阶段处理：主要修正 + orphan分配
- 输出：**16个标准relation**（15个有效 + 1个others）
- 质量：90/100分，语义一致性显著改善

#### **最终16个标准Relations**（LLM微调后）

```python
standard_relations = [
    'action_behaviors',    # 行为动作
    'additional_elements', # 附加元素（原others_relation）
    'artistic_styles',     # 艺术风格（复数）
    'character_type',      # 角色类型/表情/姿势
    'color_palette',       # 色彩搭配
    'composition_styles',  # 构图风格（复数）
    'depicted_entities',   # 描绘的实体（动物、车辆等）
    'depicted_subject',    # 描绘主体/背景
    'design_element',      # 设计元素
    'genre',               # 类型/叙事
    'lighting',            # 光照/场景
    'mood',                # 情绪氛围
    'symbolism',           # 象征意义
    'text_style',          # 文字排版
    'texture',             # 纹理/质感/对比度
    'visual_theme'         # 视觉主题（合并了dominant_color和theme）
]

relation_mapping = {
    'Color_Style': 'Color_Palette',
    'Color_Scheme': 'Color_Palette',
    'Mood_Atmosphere': 'Visual_Mood',
    'Emotional_Tone': 'Visual_Mood',
    ...
}
```

---

### **2.2 实体层聚类（每个关系下）** ✅ 方法已确定 2026-01-08

#### **Phase 2b流程**（三阶段方法）

**Stage 1: Filtering（筛选）** ✅ 已完成
- LLM判断每个entity是否属于当前relation
- 输出：keep列表 + remove列表（含suggested_relation）
- 结果：725 entities → 479 kept + 246 removed

**Stage 1.5: Redistribution（重分配）** ✅ 已完成
- 将removed entities重新分配到建议的relation
- 结果：626 unique entities，零流失

**Stage 2: Clustering + Merging（聚类+合并同义词）** 🔄 待执行
- **方法**：Embedding + LLM混合方法（借鉴Phase 2a成功经验）
- **原因**：Pure LLM方法失败（压缩率低、validation错误多）

#### **Stage 2方法：Embedding + LLM** ✅ 设计完成

```python
# scripts/phase2b_stage2_entity_clustering.py

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import hdbscan

def cluster_entities_with_embeddings(entities, method='agglomerative', min_cluster_size=2):
    """
    使用BGE Embedding + 聚类

    Args:
        entities: List[str] - 该relation下的所有entities
        method: 'agglomerative' 或 'hdbscan'

    Returns:
        clusters: Dict[int, List[str]] - {cluster_id: [entity1, entity2, ...]}
    """
    # 1. BGE Embedding
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    embeddings = model.encode(entities)

    # 2. 聚类（动态确定cluster数）
    if method == 'agglomerative':
        # 目标：压缩到约1/3
        n_clusters = max(2, min(len(entities) // 3, len(entities) // 2))
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clusterer.fit_predict(embeddings)
    else:  # hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='cosine'
        )
        labels = clusterer.fit_predict(embeddings)

    # 3. 组织clusters
    clusters = {}
    for entity, label in zip(entities, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(entity)

    return clusters


def validate_cluster_with_llm(relation, cluster_entities, api_key):
    """
    LLM验证单个cluster（简化任务）

    问题：
    1. 这些entities语义相近吗？应该合并吗？
    2. 选哪个作为canonical name？
    3. 需要split吗？
    """
    prompt = f"""
You are analyzing a cluster of entities under the relation "{relation}".

Cluster entities: {', '.join(cluster_entities)}

Questions:
1. Should these entities be merged? (yes/no)
2. If yes, which one should be the canonical name?
3. If no, should we split? How?

Response format (JSON):
{{
  "should_merge": true/false,
  "canonical_name": "entity_name",  // if should_merge=true
  "split_suggestion": null  // or split plan if should_merge=false
}}
"""

    response = call_llm_api(prompt, api_key)
    return response


def process_relation_entities(relation, entities, use_llm=True, api_key=None):
    """
    处理单个relation的entities

    流程：
    1. Embedding聚类 → 初步groups
    2. LLM验证每个cluster（可选）
    3. 输出最终merged groups
    """
    # Step 1: Embedding聚类
    clusters = cluster_entities_with_embeddings(entities, method='agglomerative')

    # Step 2: LLM验证（可选）
    final_groups = {}

    if use_llm and api_key:
        for cluster_id, cluster_entities in clusters.items():
            if len(cluster_entities) == 1:
                # 单个entity，直接保留
                final_groups[cluster_entities[0]] = cluster_entities
            else:
                # LLM验证cluster
                validation = validate_cluster_with_llm(relation, cluster_entities, api_key)

                if validation['should_merge']:
                    canonical = validation['canonical_name']
                    final_groups[canonical] = cluster_entities
                else:
                    # 不合并，保留所有entities
                    for ent in cluster_entities:
                        final_groups[ent] = [ent]
    else:
        # 不使用LLM，直接选最高频entity作为canonical name
        for cluster_id, cluster_entities in clusters.items():
            # 按字母顺序选第一个（或按频率，如果有频率信息）
            canonical = sorted(cluster_entities)[0]
            final_groups[canonical] = cluster_entities

    return final_groups
```

#### **为什么这个方法会成功？**

**Phase 2a成功证明**（Relation层）：
- 128 relations → BGE Embedding → Agglomerative(n=20) → LLM微调 → 16 relations
- 零validation错误，质量评分90/100

**优势对比**：

| 方面 | Pure LLM（失败） | Embedding + LLM（新方法） |
|------|------------------|---------------------------|
| 压缩率 | 不可控（LLM倾向保守） | 可控（embedding精确设定） |
| Validation错误 | 多（丢失entities、重复） | 无（embedding保证结构） |
| LLM任务复杂度 | 高（58个entities → 识别同义词 → 生成JSON） | 低（验证2-5个entities的cluster） |
| 成本 | 高（复杂prompt） | 低（简单验证问题） |
| 成功案例 | 无 | Phase 2a（128→16 relations） |

#### **输出**

```python
# 示例：标准化知识点词典
entity_vocabulary = {
    'Color_Palette': {
        'warm_tones': 0,
        'cool_colors': 1,
        'vibrant_saturated': 2,
        'muted_desaturated': 3,
        'monochrome': 4,
        'high_contrast': 5,
        ...                    # 最多30个
        'others': 29
    },
    'Visual_Mood': {
        'mysterious': 30,
        'joyful': 31,
        'tense_suspenseful': 32,
        'romantic': 33,
        'melancholic': 34,
        ...
        'others': 59
    },
    ...
}

# 同时保存逆向映射：entity_id -> (relation, entity_name)
id_to_knowledge = {
    0: ('Color_Palette', 'warm_tones'),
    1: ('Color_Palette', 'cool_colors'),
    ...
}
```

### **保存词典**

```python
# 保存标准化词典
save_json({
    'version': 'v1',
    'created_at': '2024-12-27',
    'statistics': {
        'num_relations': len(standard_relations),
        'num_entities': sum(len(ents) for ents in entity_vocabulary.values()),
        'max_entities_per_relation': 30
    },
    'standard_relations': standard_relations,
    'relation_mapping': relation_mapping,
    'entity_vocabulary': entity_vocabulary,
    'id_to_knowledge': id_to_knowledge
}, 'data/processed/knowledge_vocabulary_v1.json')
```

---

## ✅ Phase 3: 验证与扩充（20%数据）

### **目标**
验证词典v1的覆盖率，必要时扩充。

### **数据规模**
- **电影数量**：~800部（20%，包含最初的5%）
- **新增电影**：~600部

### **Prompt设计（限制提取）**

```
你是一个专业的电影海报视觉分析专家。请从这张电影海报中提取视觉知识点。

电影信息：
- 标题：{title}
- 类型：{genres}
- 年份：{year}

任务：从以下预定义的知识点类型中选择，提取这部电影的视觉特征。

【可选的关系类型】：
{list_of_standard_relations}

【每个关系类型下的可选实体】：
{for each relation, list top-K entities}

要求：
1. 只能从上述列表中选择
2. 每部电影最多提取10个<关系, 实体>对
3. 如果某个特征无法用现有实体描述，可以标记为<关系, "NEW_实体名称">
4. 选择最显著的特征

输出格式（JSON）：
{
  "knowledge_points": [
    {"relation": "Color_Palette", "entity": "warm_tones"},
    {"relation": "Visual_Mood", "entity": "NEW_serene_peaceful"},
    ...
  ]
}
```

### **验证逻辑**

```python
# scripts/02_validate_vocabulary.py

def validate_vocabulary(validation_results, vocabulary_v1):
    """验证词典覆盖率"""

    total_kps = 0
    new_entities = []
    unmatched_count = 0

    for movie_result in validation_results:
        for kp in movie_result['knowledge_points']:
            total_kps += 1
            relation = kp['relation']
            entity = kp['entity']

            # 检查是否是NEW实体
            if entity.startswith('NEW_'):
                new_entities.append({
                    'relation': relation,
                    'entity': entity.replace('NEW_', ''),
                    'movie_id': movie_result['movie_id']
                })
                unmatched_count += 1

    coverage_rate = (total_kps - unmatched_count) / total_kps

    print(f"总知识点数: {total_kps}")
    print(f"匹配数: {total_kps - unmatched_count}")
    print(f"未匹配数（NEW）: {unmatched_count}")
    print(f"覆盖率: {coverage_rate:.2%}")

    return coverage_rate, new_entities

def expand_vocabulary(vocabulary_v1, new_entities):
    """扩充词典"""

    # 对新实体聚类
    # ... (类似Phase 2的实体聚类)

    # 更新词典
    vocabulary_v2 = vocabulary_v1.copy()

    # 添加新的标准实体
    for relation, new_ents in new_entities_by_relation.items():
        for ent_name, ent_id in new_ents.items():
            vocabulary_v2[relation][ent_name] = ent_id

    return vocabulary_v2
```

### **决策规则**

```python
coverage_rate, new_entities = validate_vocabulary(results, vocab_v1)

if coverage_rate >= 0.90:
    print("✓ 词典覆盖率足够，使用v1进行全量提取")
    final_vocabulary = vocab_v1
else:
    print(f"⚠ 词典覆盖率较低（{coverage_rate:.2%}），需要扩充")

    # 分析新实体
    print(f"发现 {len(new_entities)} 个新实体")

    # 扩充词典
    final_vocabulary = expand_vocabulary(vocab_v1, new_entities)

    print(f"✓ 词典已扩充：v1 → v2")
```

### **成本估算**

- 800张图 × $0.0003/图 = **$0.24**

---

## 🚀 Phase 4: 全量提取（80%数据）

### **数据规模**
- **电影数量**：~3100部（剩余80%）
- **使用词典**：v2（如果需要扩充）或v1

### **Prompt**
与Phase 3相同（限制提取），但不允许NEW标记。

### **实现**

```python
# scripts/03_extract_knowledge_full.py

def extract_full_dataset(final_vocabulary):
    """全量提取"""

    all_movies = load_all_movies_with_posters()
    already_extracted = load_already_extracted()  # Phase 1-3已提取的

    remaining_movies = [m for m in all_movies if m.id not in already_extracted]

    print(f"Extracting knowledge from {len(remaining_movies)} movies...")

    results = []
    for movie in tqdm(remaining_movies):
        knowledge = extract_knowledge_with_vocabulary(
            poster_path=f"data/raw/ml-1m/posters/{movie.id}.jpg",
            movie_info=movie,
            vocabulary=final_vocabulary
        )
        results.append({
            'movie_id': movie.id,
            'knowledge_points': knowledge
        })

    # 合并所有结果
    all_results = already_extracted + results

    save_json(all_results, 'data/processed/movie_knowledge_full.json')

    print(f"✓ Total movies with knowledge: {len(all_results)}")
```

### **成本估算**

- 3100张图 × $0.0003/图 = **$0.93**

### **总成本（Phase 1-4）**

- Phase 1: $0.06
- Phase 2: (本地聚类，无成本)
- Phase 3: $0.24
- Phase 4: $0.93
- **总计**: **~$1.23**

---

## 👤 Phase 5: 动态用户兴趣提取

### **目标**
基于用户历史观影，提取其视觉知识偏好。

### **方法A：频率聚合（基础）**

```python
# src/extraction/user_extractor.py

def extract_user_interests_frequency(user_id, history, movie_knowledge):
    """
    基于频率的用户兴趣提取

    Args:
        user_id: 用户ID
        history: 用户历史[(movie_id, rating, timestamp), ...]
        movie_knowledge: {movie_id: [knowledge_points]}

    Returns:
        user_interests: {
            'Color_Palette': {entity_id: weight, ...},
            'Visual_Mood': {...},
            ...
        }
    """

    # 1. 收集用户看过的所有电影的知识点
    knowledge_counts = {}

    for movie_id, rating, timestamp in history:
        if movie_id not in movie_knowledge:
            continue

        # 权重：高分电影权重更高
        weight = (rating - 3.0) / 2.0  # 归一化到[-1, 1]
        if weight < 0:
            continue  # 忽略低分电影

        for kp in movie_knowledge[movie_id]:
            relation = kp['relation']
            entity_id = kp['entity_id']

            if relation not in knowledge_counts:
                knowledge_counts[relation] = {}
            if entity_id not in knowledge_counts[relation]:
                knowledge_counts[relation][entity_id] = 0

            knowledge_counts[relation][entity_id] += weight

    # 2. 归一化
    user_interests = {}
    for relation, entities in knowledge_counts.items():
        total = sum(entities.values())
        user_interests[relation] = {
            ent_id: count / total
            for ent_id, count in entities.items()
        }

    return user_interests
```

### **方法B：TF-IDF加权（改进）**

```python
def extract_user_interests_tfidf(user_id, history, movie_knowledge, all_users):
    """
    TF-IDF加权：强调用户独特的偏好

    TF：用户对该知识点的偏好程度
    IDF：知识点的区分度（越少人喜欢，越有区分度）
    """

    # 1. TF：用户频率
    user_freq = compute_user_knowledge_frequency(user_id, history, movie_knowledge)

    # 2. IDF：全局频率
    knowledge_idf = compute_knowledge_idf(all_users, movie_knowledge)

    # 3. TF-IDF
    user_interests = {}
    for relation, entities in user_freq.items():
        user_interests[relation] = {}
        for ent_id, tf in entities.items():
            idf = knowledge_idf.get(ent_id, 1.0)
            user_interests[relation][ent_id] = tf * idf

    return user_interests
```

### **输出**

```python
# 示例：用户兴趣表示
user_interests = {
    'user_id': 123,
    'interests': {
        'Color_Palette': {
            0: 0.35,   # warm_tones
            2: 0.25,   # vibrant_saturated
            5: 0.15,   # high_contrast
            ...
        },
        'Visual_Mood': {
            30: 0.40,  # mysterious
            32: 0.30,  # tense_suspenseful
            ...
        },
        ...
    },
    'statistics': {
        'num_movies_watched': 150,
        'num_knowledge_preferences': 45,
        'top_3_relations': ['Visual_Mood', 'Genre_Visual', 'Color_Palette']
    }
}
```

---

## 🕸️ Phase 6: 知识图谱构建

### **图谱结构**

```
User节点 ←→ Knowledge节点 ←→ Item节点

节点类型：
- User: 6040个用户（5-core后）
- Knowledge: ~500-1000个知识点（15类关系 × ~30实体）
- Item: ~3700部电影（5-core后）

边类型：
- User -[interested_in]-> Knowledge  (权重 = 用户兴趣分数)
- Knowledge -[describes]-> Item       (权重 = 1.0)
- User -[rated]-> Item                (权重 = rating/5.0，保留原始交互)
```

### **构建逻辑**

```python
# src/graph/builder.py

import networkx as nx
import torch
from torch_geometric.data import HeteroData

def build_knowledge_graph(
    users,
    items,
    user_interests,
    movie_knowledge,
    ratings
):
    """
    构建User-Knowledge-Item异构图

    Returns:
        graph: HeteroData对象（PyG格式）
    """

    graph = HeteroData()

    # 1. 添加节点
    graph['user'].num_nodes = len(users)
    graph['knowledge'].num_nodes = num_total_knowledge_points
    graph['item'].num_nodes = len(items)

    # 2. User -> Knowledge边
    user_know_edges = []
    user_know_weights = []

    for user_id, interests in user_interests.items():
        for relation, entities in interests['interests'].items():
            for entity_id, weight in entities.items():
                if weight < 0.05:  # 过滤低权重边
                    continue
                user_know_edges.append([user_id, entity_id])
                user_know_weights.append(weight)

    graph['user', 'interested_in', 'knowledge'].edge_index = torch.tensor(
        user_know_edges, dtype=torch.long
    ).t().contiguous()
    graph['user', 'interested_in', 'knowledge'].edge_attr = torch.tensor(
        user_know_weights, dtype=torch.float
    )

    # 3. Knowledge -> Item边
    know_item_edges = []

    for item_id, kps in movie_knowledge.items():
        for kp in kps:
            entity_id = kp['entity_id']
            # 过滤others
            if is_others(entity_id):
                continue  # 不连接others
            know_item_edges.append([entity_id, item_id])

    graph['knowledge', 'describes', 'item'].edge_index = torch.tensor(
        know_item_edges, dtype=torch.long
    ).t().contiguous()

    # 4. User -> Item边（原始交互，保留）
    user_item_edges = []
    user_item_ratings = []

    for user_id, item_id, rating in ratings:
        user_item_edges.append([user_id, item_id])
        user_item_ratings.append(rating / 5.0)

    graph['user', 'rated', 'item'].edge_index = torch.tensor(
        user_item_edges, dtype=torch.long
    ).t().contiguous()
    graph['user', 'rated', 'item'].edge_attr = torch.tensor(
        user_item_ratings, dtype=torch.float
    )

    return graph
```

### **Others节点处理**

```python
def is_others(entity_id, vocabulary):
    """判断是否是others实体"""
    for relation, entities in vocabulary.items():
        if entity_id == entities.get('others'):
            return True
    return False

# 构建图谱时：
# 方案：不连接others节点
if is_others(entity_id):
    continue  # 跳过
```

---

## 🧠 Phase 7: 带Mask机制的推荐模型

### **模型架构**

```python
# src/model/ours_full.py

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv

class OursFull(nn.Module):
    def __init__(self, num_users, num_items, num_knowledge, embed_dim=64):
        super().__init__()

        # 节点embedding
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.knowledge_embed = nn.Embedding(num_knowledge, embed_dim)

        # 异构图卷积层
        self.conv1 = HeteroConv({
            ('user', 'interested_in', 'knowledge'): GATConv(embed_dim, embed_dim),
            ('knowledge', 'describes', 'item'): GATConv(embed_dim, embed_dim),
            ('user', 'rated', 'item'): GATConv(embed_dim, embed_dim),
        })

        self.conv2 = HeteroConv({...})  # 第二层

        # Mask机制
        self.knowledge_mask = None  # 在训练/推理时设置

    def forward(self, graph, user_ids, item_ids):
        # 1. 初始embedding
        x_dict = {
            'user': self.user_embed.weight,
            'item': self.item_embed.weight,
            'knowledge': self.knowledge_embed.weight
        }

        # 2. Mask knowledge节点
        if self.knowledge_mask is not None:
            x_dict['knowledge'] = x_dict['knowledge'] * self.knowledge_mask.unsqueeze(1)

        # 3. 图卷积
        x_dict = self.conv1(x_dict, graph.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, graph.edge_index_dict)

        # 4. 预测
        user_emb = x_dict['user'][user_ids]
        item_emb = x_dict['item'][item_ids]

        scores = (user_emb * item_emb).sum(dim=1)

        return scores
```

### **Mask策略**

```python
# src/model/mask_strategy.py

def create_knowledge_mask(vocabulary, mode='train'):
    """
    创建知识点mask

    Args:
        vocabulary: 知识点词典
        mode: 'train' or 'inference'

    Returns:
        mask: Tensor of shape [num_knowledge]
              1.0 = keep, 0.0 = mask
    """

    num_knowledge = sum(len(ents) for ents in vocabulary.values())
    mask = torch.ones(num_knowledge)

    for relation, entities in vocabulary.items():
        for ent_name, ent_id in entities.items():
            if ent_name == 'others':
                if mode == 'train':
                    mask[ent_id] = 0.5  # 训练时降权
                else:
                    mask[ent_id] = 0.0  # 推理时完全mask

    return mask

# 训练时
model.knowledge_mask = create_knowledge_mask(vocab, mode='train')

# 推理时
model.knowledge_mask = create_knowledge_mask(vocab, mode='inference')
```

---

## 📂 文件组织

```
data/
├── processed/
│   ├── knowledge_pilot_raw.json          # Phase 1输出
│   ├── knowledge_vocabulary_v1.json      # Phase 2输出
│   ├── knowledge_vocabulary_v2.json      # Phase 3输出（如果扩充）
│   ├── knowledge_validation.json         # Phase 3验证结果
│   ├── movie_knowledge_full.json         # Phase 4全量知识点
│   └── user_interests.json               # Phase 5用户兴趣
│
└── graphs/
    └── knowledge_graph.pt                # Phase 6图谱

scripts/
├── 01_extract_knowledge_pilot.py         # Phase 1
├── 02_cluster_knowledge.py               # Phase 2
├── 03_validate_vocabulary.py             # Phase 3
├── 04_extract_knowledge_full.py          # Phase 4
├── 05_extract_user_interests.py          # Phase 5
└── 06_build_graph.py                     # Phase 6

src/
├── extraction/
│   ├── llm_client.py                     # MLLM API客户端
│   ├── movie_extractor.py                # 电影知识提取
│   ├── user_extractor.py                 # 用户兴趣提取
│   └── prompt_templates.py               # Prompt模板
│
├── clustering/
│   ├── relation_clusterer.py             # 关系层聚类
│   └── entity_clusterer.py               # 实体层聚类
│
├── graph/
│   └── builder.py                        # 图谱构建
│
└── model/
    ├── ours_full.py                      # 完整模型
    ├── ours_visual.py                    # 消融：只用视觉知识
    ├── ours_interest.py                  # 消融：只用兴趣
    └── mask_strategy.py                  # Mask机制
```

---

## 📊 数据规模统计

### **知识点规模**

| 指标 | Phase 1 实际 (170部电影) | Phase 3 (预期) | Phase 4 (预期) |
|------|--------------|---------------|----------------|
| 电影数 | **170** ✅ | 800 | 3900 |
| 总知识点数 | **1869** ✅ | ~8000 | ~39000 |
| 唯一关系数 | **128** → **14** ✅ | - | - |
| 唯一实体数 | **626** → **200-300** (待聚类) | **250-350** | **250-350** |

### **词典规模**（基于实际结果）

**Relation层**（✅ 已完成 2026-01-03）：
- **关系类数**：**16个**（15个有效 + 1个others）
- 方法：Agglomerative N=20 + LLM增量微调
- 覆盖率：100%
- 质量评分：90/100
- 保存文件：`relation_mapping_final_v2.json`

**Entity层**（🔄 待进行）：
- **实体数**：每类最多20-30 → 总计 **16×25 = 400** + 16个others = **~416个**（预期）

### **图谱规模**（预期）

| 节点类型 | 数量（预期） |
|----------|------|
| User | 6040 (5-core后) |
| Knowledge | ~416 (16 relations × 26 entities/relation 平均) |
| Item | 3700 (5-core后) |
| **总节点** | **~10156** |

| 边类型 | 数量（估算） |
|--------|--------------|
| User → Knowledge | ~6040 × 15-20 = 90-120K |
| Knowledge → Item | ~3700 × 5 = 18.5K |
| User → Item | ~900K (5-core后) |
| **总边数** | **~1M** |

---

## 💰 总成本估算

| Phase | 电影数 | 单价 | 成本 |
|-------|--------|------|------|
| Phase 1 (5%探索) | 200 | $0.0003 | $0.06 |
| Phase 3 (20%验证) | 800 | $0.0003 | $0.24 |
| Phase 4 (80%全量) | 3100 | $0.0003 | $0.93 |
| **总计** | **4100** | - | **$1.23** |

**备注**：
- 使用GPT-4o-mini（$0.0003/图）
- 不包括Qwen3-VL（本地部署，无API成本）
- Phase 2（聚类）本地计算，无成本

---

## 🎯 关键创新点总结

### **1. 双阶段知识点提取**
- 小规模自由探索 → 数据驱动发现知识类型
- 验证扩充 → 保证词典覆盖率
- 限制提取 → 减少LLM幻觉和不一致性

### **2. 分层知识标准化**
- 关系层聚类：将50-100种原始关系类型 → 10-15种标准关系
- 实体层聚类：每个关系下500-1000种实体 → 最多30种标准实体
- Others机制：优雅处理长尾低频知识点

### **3. 动态用户兴趣提取**
- 基于历史观影的知识偏好聚合
- TF-IDF加权强调用户独特性
- 多维度兴趣表示

### **4. Mask增强学习**
- 训练时降权others知识点
- 推理时完全mask others
- 提升模型鲁棒性，减少对低质量知识点的依赖

---

## 📝 下一步行动

### **当前阶段：Phase 3 验证与扩充** ⬅️ 我们在这里

**🎉 Phase 2 已完成！**

**Phase 2a（Relation层标准化）**：✅ 100%完成
- 128个原始relations → 16个标准relations
- 方法：BGE Embedding + Agglomerative + LLM增量微调
- 质量：90/100分，覆盖率100%

**Phase 2b（Entity层标准化）**：✅ 100%完成
- **Stage 1**: Filtering（725 → 479 kept + 246 removed）
- **Stage 1.5**: Redistribution（626 unique entities，零流失）
- **Stage 2**: BERTopic聚类（626 → 133 canonical clusters）
- **Stage 3**: LLM三步Refinement（133 → 178 clusters）
  - 关键创新：Split→Merge→Rename + CoT + Business Context
  - 成功修复：superhero → "hero"（不再是villain）
  - 最终：165个标准entities + 13个噪声entities
- **Vocabulary提取**：`results/standard_entity_vocabulary.json`

---

### **Phase 3：验证与扩充（下一步）**

**目标**：验证vocabulary v1的覆盖率，决定是否扩充

**数据规模**：
- 20%数据（约800部电影，包含Phase 1的170部）
- 新增约630部电影

**任务列表**：

1. ⏸️ **设计限制提取prompt**
   - 提供15个standard relations
   - 提供每个relation下的标准entities（165个）
   - LLM只能从列表中选择
   - 如果无法匹配，标记为`NEW_entity_name`
   - 每部电影最多10个知识点

2. ⏸️ **实现Phase 3提取脚本**
   - `scripts/phase3_validate_vocabulary.py`
   - 加载`standard_entity_vocabulary.json`
   - 构建限制提取prompt
   - 提取约630部新电影
   - 统计NEW标记的比例

3. ⏸️ **统计覆盖率**
   - 覆盖率 = (匹配数 / 总知识点数)
   - 目标：≥90%

4. ⏸️ **决策分支**
   - **如果覆盖率 ≥90%**：
     - vocabulary v1足够
     - 直接进入Phase 4全量提取

   - **如果覆盖率 <90%**：
     - 收集所有NEW entities
     - 对NEW entities聚类（类似Phase 2b Stage 2-3）
     - 扩充vocabulary v1 → v2
     - 更新`standard_entity_vocabulary.json`

5. ⏸️ **保存最终vocabulary**
   - 如果扩充：`standard_entity_vocabulary_v2.json`
   - 预期最终规模：200-250个标准entities

---

### **后续阶段（Phase 4-7）**

**Phase 4**：全量提取（80%数据）
- 约3100部电影
- 使用最终vocabulary（v1或v2）
- 限制提取（不允许NEW标记）
- 合并Phase 1-4所有结果

**Phase 5**：用户兴趣提取
- 基于历史观影聚合知识偏好
- TF-IDF加权
- 生成用户兴趣表示

**Phase 6**：知识图谱构建
- User-Knowledge-Item异构图
- 边类型：interested_in, describes, rated
- Mask机制：抑制others知识点

**Phase 7**：推荐模型训练
- 带Mask机制的GNN
- Baseline对比（13个方法）
- 消融实验

---

### **关键里程碑**

- ✅ **Phase 1完成**（2026-01-02）：170部电影，1869个知识点
- ✅ **Phase 2a完成**（2026-01-03）：16个标准relations
- ✅ **Phase 2b完成**（2026-01-11）：165个标准entities 🎉
- ⏸️ **Phase 3开始**（2026-01-11+）：验证与扩充
- ⏸️ **1月13日目标**：提供KG数据给学弟跑baseline
- ⏸️ **2月9日KDD deadline**

---

*文档结束*
*最后更新：2026-01-11*
