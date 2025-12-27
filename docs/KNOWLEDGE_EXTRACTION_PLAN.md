# LLM知识抽取方案

*Created: 2024-12-27*
*Status: 设计阶段*

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

### **2.1 关系层聚类**

#### **输入**
从Phase 1收集所有唯一的"关系"名称（如：`Color_Palette`, `Color_Style`, `Visual_Mood`, `Mood_Atmosphere`...）

#### **方法**

```python
# src/clustering/relation_clusterer.py

from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from collections import Counter

def cluster_relations(raw_relations, max_clusters=15):
    """
    聚类关系类型

    Args:
        raw_relations: List of relation names
        max_clusters: 最多保留的关系类数量

    Returns:
        relation_mapping: {原始关系 -> 标准关系}
        standard_relations: List of 标准关系名
    """

    # 1. 统计频率
    relation_freq = Counter(raw_relations)

    # 2. BGE embedding
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    embeddings = model.encode(list(relation_freq.keys()))

    # 3. HDBSCAN聚类
    clusterer = HDBSCAN(
        min_cluster_size=3,  # 至少3个关系归为一类
        metric='cosine',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # 4. 提取标准关系名（每个cluster选代表）
    clusters = {}
    for rel, label in zip(relation_freq.keys(), cluster_labels):
        if label == -1:  # noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((rel, relation_freq[rel]))

    # 5. 每个cluster选最高频的作为标准名
    standard_relations = []
    relation_mapping = {}

    for cluster_id, relations in clusters.items():
        # 按频率排序
        relations.sort(key=lambda x: x[1], reverse=True)
        standard_name = relations[0][0]  # 最高频

        standard_relations.append(standard_name)

        # 建立映射
        for rel, _ in relations:
            relation_mapping[rel] = standard_name

    # 6. 如果cluster太多，合并小cluster
    if len(standard_relations) > max_clusters:
        standard_relations = merge_small_clusters(
            standard_relations,
            max_clusters
        )

    # 7. Noise和低频关系 → Others
    relation_mapping['Others_Relation'] = 'Others_Relation'
    for rel in raw_relations:
        if rel not in relation_mapping:
            relation_mapping[rel] = 'Others_Relation'

    return relation_mapping, standard_relations + ['Others_Relation']
```

#### **输出**

```python
# 示例输出
standard_relations = [
    'Color_Palette',       # 整合了 Color_Style, Color_Scheme等
    'Visual_Mood',         # 整合了 Mood_Atmosphere, Emotional_Tone等
    'Composition_Style',   # 整合了 Layout, Composition等
    'Lighting_Effect',
    'Era_Aesthetic',
    'Genre_Visual',
    'Character_Display',
    'Setting_Environment',
    'Poster_Design',
    'Visual_Symbolism',
    ...                    # 最多15个
    'Others_Relation'      # 低频关系
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

### **2.2 实体层聚类（每个关系下）**

#### **方法**

```python
# src/clustering/entity_clusterer.py

def cluster_entities_per_relation(
    knowledge_points,
    relation_mapping,
    max_entities_per_relation=30
):
    """
    对每个标准关系，聚类其下的实体

    Returns:
        entity_vocabulary: {
            'Color_Palette': {
                'warm_tones': 23,
                'cool_colors': 45,
                ...
                'others': 999
            },
            'Visual_Mood': {...},
            ...
        }
    """

    # 1. 按标准关系分组实体
    relation_entities = {}
    for kp in knowledge_points:
        std_rel = relation_mapping.get(kp['relation'], 'Others_Relation')
        entity = kp['entity']

        if std_rel not in relation_entities:
            relation_entities[std_rel] = []
        relation_entities[std_rel].append(entity)

    # 2. 对每个关系，聚类其实体
    entity_vocabulary = {}
    entity_id_counter = 0

    for relation, entities in relation_entities.items():
        # 统计频率
        entity_freq = Counter(entities)

        # BGE embedding
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        embeddings = model.encode(list(entity_freq.keys()))

        # HDBSCAN聚类
        clusterer = HDBSCAN(min_cluster_size=2, metric='cosine')
        labels = clusterer.fit_predict(embeddings)

        # 提取标准实体
        clusters = {}
        for ent, label in zip(entity_freq.keys(), labels):
            if label == -1:
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((ent, entity_freq[ent]))

        # 每个cluster选代表
        standard_entities = {}
        for cluster_id, ents in clusters.items():
            ents.sort(key=lambda x: x[1], reverse=True)
            standard_name = ents[0][0]
            standard_entities[standard_name] = entity_id_counter
            entity_id_counter += 1

        # 限制数量
        if len(standard_entities) > max_entities_per_relation:
            # 保留top-K高频实体
            top_entities = dict(sorted(
                standard_entities.items(),
                key=lambda x: entity_freq[x[0]],
                reverse=True
            )[:max_entities_per_relation])
            standard_entities = top_entities

        # 添加others
        standard_entities['others'] = entity_id_counter
        entity_id_counter += 1

        entity_vocabulary[relation] = standard_entities

    return entity_vocabulary
```

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

## 📊 预期统计

### **知识点规模**

| 指标 | Phase 1 (5%) | Phase 3 (20%) | Phase 4 (100%) |
|------|--------------|---------------|----------------|
| 电影数 | 200 | 800 | 3900 |
| 总知识点数 | ~2000 | ~8000 | ~39000 |
| 唯一关系数 | 50-100 → **10-15** | - | - |
| 唯一实体数 | 500-1000 → **300-450** | **350-500** | **350-500** |

### **词典规模**

- **关系类数**：10-15 + 1(others) = **11-16类**
- **实体数**：每类最多30 → 总计 **15×30 = 450** + 15个others = **~465个**

### **图谱规模**

| 节点类型 | 数量 |
|----------|------|
| User | 6040 (5-core后) |
| Knowledge | ~465 |
| Item | 3700 (5-core后) |
| **总节点** | **~10205** |

| 边类型 | 数量（估算） |
|--------|--------------|
| User → Knowledge | ~6040 × 20 = 120K |
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

1. **实现Phase 1**：小规模探索脚本
2. **实现Phase 2**：双层聚类脚本
3. **实现Phase 3**：验证与扩充
4. **实现Phase 4-7**：全量提取→用户兴趣→图谱→模型

---

*文档结束*
