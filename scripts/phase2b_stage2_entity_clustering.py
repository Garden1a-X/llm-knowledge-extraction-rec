#!/usr/bin/env python3
"""
Phase 2b Stage 2: Entity聚类（Embedding + LLM微调）

基于Phase 2a成功经验：
1. 使用BGE embedding + 聚类算法进行初步聚类
2. 使用LLM微调聚类结果（验证语义、选择canonical name）

流程：
- 对每个relation独立处理
- Embedding聚类 → 自动确定合理的cluster数
- LLM验证并微调每个cluster
- 输出最终的merged groups

使用：
  python scripts/phase2b_stage2_entity_clustering.py

输入：
  results/entity_redistribution_stage1_5_redistributed.json (Stage 1.5输出)

输出：
  results/entity_redistribution_stage2_clustered.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# 导入共享函数
from scripts.phase2b_shared import RELATION_DEFINITIONS, call_gpt4


def cluster_entities_with_embeddings(
    entities: List[str],
    method: str = 'hdbscan',
    min_cluster_size: int = 2,
    distance_threshold: float = 0.3
) -> Tuple[List[int], np.ndarray]:
    """
    使用Embedding进行实体聚类

    Args:
        entities: entity列表
        method: 'hdbscan' or 'agglomerative'
        min_cluster_size: HDBSCAN最小cluster大小
        distance_threshold: Agglomerative距离阈值

    Returns:
        (cluster_labels, embeddings)
    """
    from sentence_transformers import SentenceTransformer

    # 生成embeddings
    print(f"  生成embeddings（BGE-base-en-v1.5）...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    embeddings = model.encode(entities, show_progress_bar=False)

    # 聚类
    if method == 'hdbscan':
        try:
            import hdbscan
            print(f"  使用HDBSCAN聚类（min_cluster_size={min_cluster_size}）...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='cosine',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embeddings)
        except ImportError:
            print("  ⚠️  HDBSCAN未安装，使用Agglomerative...")
            method = 'agglomerative'

    if method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering

        # 动态确定cluster数：至少减少到1/3，但不超过原数量的1/2
        n_clusters = max(2, min(len(entities) // 3, len(entities) // 2))

        print(f"  使用Agglomerative聚类（n_clusters={n_clusters}）...")
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clusterer.fit_predict(embeddings)

    return labels, embeddings


def build_cluster_validation_prompt(
    relation: str,
    cluster_entities: List[str]
) -> str:
    """
    构建cluster验证prompt（简化版，不要求生成完整JSON）

    Args:
        relation: relation名称
        cluster_entities: cluster中的entities

    Returns:
        prompt字符串
    """
    rel_def = RELATION_DEFINITIONS.get(relation, {})

    prompt = f"""你是一个专业的知识分类专家。

## 任务

embedding聚类算法将以下entities聚在了一起，请验证它们是否语义相近，应该合并。

**Relation**: {relation}
**定义**: {rel_def.get('description', '')}

**Cluster包含的entities**:
"""

    for i, entity in enumerate(cluster_entities, 1):
        prompt += f"{i}. {entity}\n"

    prompt += f"""

## 问题

1. **这{len(cluster_entities)}个entities语义相近吗？应该合并成一个标准entity吗？**
   - 如果是真正的同义词或非常相近的概念 → 应该合并
   - 如果语义差异较大，只是embedding相似 → 不应该合并

2. **如果应该合并，选择哪个entity作为canonical name（代表名）？**
   - ⚠️ **CRITICAL**: canonical name **必须**从上面的列表中选择（编号1-{len(cluster_entities)}）
   - 优先选择更通用、更短、更标准的名称
   - **绝对不能**创造新名字

3. **如果不应该合并，建议如何split？**
   - 说明哪些entities应该分成不同的groups

## 输出格式

请以JSON格式输出：

```json
{{
  "should_merge": true/false,
  "canonical_name": "entity_name",  // 必须从上面列表中选择！
  "members": ["entity1", "entity2", ...],  // 如果should_merge=true
  "reason": "简短说明",
  "split_suggestion": [  // 如果should_merge=false
    {{"canonical": "entity1", "members": ["entity1", "entity2"]}},
    {{"canonical": "entity3", "members": ["entity3"]}}
  ]
}}
```

**重要规则**：
- canonical_name **必须**是上面编号1-{len(cluster_entities)}中的某个entity
- 不要创造新名字
- members数组必须包含canonical_name自己

请分析并输出JSON结果。
"""

    return prompt


def validate_and_refine_cluster(
    relation: str,
    cluster_entities: List[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用LLM验证并微调单个cluster

    Returns:
        validation结果
    """
    if len(cluster_entities) == 1:
        # 单个entity，直接返回
        return {
            'should_merge': True,
            'canonical_name': cluster_entities[0],
            'members': cluster_entities,
            'reason': 'Only one entity in cluster'
        }

    # 构建prompt
    prompt = build_cluster_validation_prompt(relation, cluster_entities)

    # 调用GPT-4
    try:
        result = call_gpt4(prompt, api_key=api_key, base_url=base_url)

        # 验证canonical_name
        if result.get('should_merge', False):
            canonical = result.get('canonical_name')
            if canonical not in cluster_entities:
                print(f"    ⚠️  LLM创造了新名字'{canonical}'，使用第一个entity代替")
                result['canonical_name'] = cluster_entities[0]

        return result

    except Exception as e:
        print(f"    ⚠️  LLM调用失败: {e}")
        # 失败时保守策略：使用第一个entity作为代表
        return {
            'should_merge': True,
            'canonical_name': cluster_entities[0],
            'members': cluster_entities,
            'reason': f'LLM failed, using conservative merge: {e}'
        }


def process_relation_clustering(
    relation: str,
    entities: List[str],
    clustering_method: str = 'agglomerative',
    use_llm: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    处理单个relation的entity聚类

    Args:
        relation: relation名称
        entities: entity列表
        clustering_method: 'hdbscan' or 'agglomerative'
        use_llm: 是否使用LLM验证
        api_key: OpenAI API key
        base_url: API base URL

    Returns:
        聚类结果
    """
    print(f"\n{'='*80}")
    print(f"处理Relation: {relation}")
    print(f"{'='*80}")
    print(f"Entities数量: {len(entities)}")

    if len(entities) <= 3:
        print(f"  ⚠️  Entity数量太少（<= 3），跳过聚类")
        # 直接返回，不聚类
        merged_groups = {entity: [entity] for entity in entities}
        return {
            'relation': relation,
            'original_count': len(entities),
            'clustering_method': 'none',
            'clusters_before_llm': len(entities),
            'groups_after_llm': len(entities),
            'merged_groups': merged_groups,
            'compression_ratio': 1.0
        }

    # Step 1: Embedding聚类
    print(f"\n[Step 1] Embedding聚类")
    labels, embeddings = cluster_entities_with_embeddings(
        entities,
        method=clustering_method,
        min_cluster_size=2
    )

    # 统计clusters
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = sum(1 for l in labels if l == -1)

    print(f"  ✓ 聚类完成:")
    print(f"    Clusters: {n_clusters}")
    if n_noise > 0:
        print(f"    Noise points (label=-1): {n_noise}")

    # 组织clusters
    clusters = defaultdict(list)
    for entity, label in zip(entities, labels):
        clusters[label].append(entity)

    # Step 2: LLM验证和微调（可选）
    merged_groups = {}

    if use_llm:
        print(f"\n[Step 2] LLM验证和微调")
        print(f"  处理{len(clusters)}个clusters...")

        for cluster_id, cluster_entities in clusters.items():
            cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
            print(f"\n  {cluster_name}: {len(cluster_entities)} entities")
            print(f"    Entities: {cluster_entities[:5]}{'...' if len(cluster_entities) > 5 else ''}")

            # LLM验证
            validation = validate_and_refine_cluster(
                relation,
                cluster_entities,
                api_key=api_key,
                base_url=base_url
            )

            if validation['should_merge']:
                canonical = validation['canonical_name']
                members = validation.get('members', cluster_entities)
                print(f"    ✓ 合并为: {canonical}")
                merged_groups[canonical] = members
            else:
                # LLM建议split
                print(f"    ⚠️  LLM建议split")
                splits = validation.get('split_suggestion', [])
                if splits:
                    for split_group in splits:
                        canonical = split_group['canonical']
                        members = split_group['members']
                        merged_groups[canonical] = members
                else:
                    # 没有split建议，保守地每个entity独立
                    for entity in cluster_entities:
                        merged_groups[entity] = [entity]
    else:
        # 不使用LLM，直接选择第一个entity作为代表
        print(f"\n[Step 2] 跳过LLM验证")
        for cluster_id, cluster_entities in clusters.items():
            canonical = cluster_entities[0]
            merged_groups[canonical] = cluster_entities

    # 统计
    compression_ratio = len(merged_groups) / len(entities)

    print(f"\n{'='*80}")
    print(f"聚类完成:")
    print(f"  原始entities: {len(entities)}")
    print(f"  Embedding clusters: {n_clusters}")
    print(f"  最终groups: {len(merged_groups)}")
    print(f"  压缩率: {compression_ratio:.2%}")
    print(f"{'='*80}")

    return {
        'relation': relation,
        'original_count': len(entities),
        'clustering_method': clustering_method,
        'clusters_before_llm': n_clusters,
        'groups_after_llm': len(merged_groups),
        'merged_groups': merged_groups,
        'compression_ratio': compression_ratio
    }


def main():
    print("="*80)
    print("Phase 2b Stage 2: Entity聚类（Embedding + LLM微调）")
    print("="*80)
    print()

    # 输入文件
    default_input = 'results/entity_redistribution_stage1_5_redistributed.json'
    input_file = input(f"输入文件 (default: {default_input}): ").strip() or default_input

    input_path = PROJECT_ROOT / input_file

    if not input_path.exists():
        print(f"❌ 错误: 文件不存在: {input_path}")
        return

    # 加载数据
    print(f"\n加载输入文件: {input_path}")
    with open(input_path, 'r') as f:
        stage1_5_data = json.load(f)

    results = stage1_5_data.get('results', {})

    print(f"找到 {len(results)} 个relations")

    # 交互式配置
    print(f"\n{'='*80}")
    print("配置参数")
    print(f"{'='*80}")

    # 选择要处理的relation
    print("\n请选择要处理的relation:")
    print("  - 输入relation名称（如 visual_theme）")
    print("  - 输入 'all' 处理所有relations")

    relation_input = input("\nRelation: ").strip()

    if relation_input == 'all':
        process_all = True
        selected_relation = None
    elif relation_input in results:
        process_all = False
        selected_relation = relation_input
    else:
        print(f"❌ 错误: Relation '{relation_input}' 不存在")
        return

    # 聚类方法
    clustering_method = input("\n聚类方法 (hdbscan/agglomerative, default: agglomerative): ").strip() or 'agglomerative'

    # 是否使用LLM
    use_llm_input = input("\n使用LLM验证? (y/n, default: y): ").strip().lower()
    use_llm = use_llm_input != 'n'

    # LLM配置
    api_key = None
    base_url = None

    if use_llm:
        print("\n请配置LLM参数:")
        api_key = input("  API Key (optional): ").strip() or None
        base_url = input("  Base URL (optional): ").strip() or None

    # 输出文件
    default_output = 'results/entity_redistribution_stage2_clustered.json'
    output_file = input(f"\n输出文件 (default: {default_output}): ").strip() or default_output

    # 确认
    print(f"\n{'='*80}")
    print("配置总结")
    print(f"{'='*80}")
    print(f"  处理范围: {'所有relations' if process_all else selected_relation}")
    print(f"  聚类方法: {clustering_method}")
    print(f"  使用LLM: {'是' if use_llm else '否'}")
    if use_llm:
        print(f"  API Key: {'已设置' if api_key else '使用环境变量'}")
        print(f"  Base URL: {base_url if base_url else '默认'}")

    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 处理relations
    all_results = {}

    if process_all:
        for relation, rel_data in sorted(results.items()):
            entities = rel_data.get('final_entities', [])

            result = process_relation_clustering(
                relation,
                entities,
                clustering_method=clustering_method,
                use_llm=use_llm,
                api_key=api_key,
                base_url=base_url
            )

            all_results[relation] = result
    else:
        entities = results[selected_relation].get('final_entities', [])

        result = process_relation_clustering(
            selected_relation,
            entities,
            clustering_method=clustering_method,
            use_llm=use_llm,
            api_key=api_key,
            base_url=base_url
        )

        all_results[selected_relation] = result

    # 保存结果
    output_path = PROJECT_ROOT / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'stage': 'stage2_clustered',
            'clustering_method': clustering_method,
            'use_llm_validation': use_llm,
            'relations_processed': list(all_results.keys()),
            'stage1_5_input': str(input_path)
        },
        'results': all_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 总结
    print(f"\n{'='*80}")
    print("✅ Stage 2 完成!")
    print(f"{'='*80}")
    print(f"结果已保存到: {output_path}")

    print("\n统计:")
    total_original = 0
    total_final = 0

    for relation, result in all_results.items():
        original = result['original_count']
        final = result['groups_after_llm']
        compression = result['compression_ratio']

        print(f"  {relation}:")
        print(f"    {original} → {final} groups (压缩率: {compression:.1%})")

        total_original += original
        total_final += final

    overall_compression = total_final / total_original if total_original > 0 else 1.0
    print(f"\n  总计: {total_original} → {total_final} groups (压缩率: {overall_compression:.1%})")

    print("\n下一步:")
    print("  1. 检查聚类结果")
    print("  2. 继续处理其他relations或进行下一阶段")


if __name__ == '__main__':
    main()
