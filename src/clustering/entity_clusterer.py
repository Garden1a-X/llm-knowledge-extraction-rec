"""
Entity聚类模块 - Phase 2b Stage 2
将Stage 1.5重分配后的entities按relation聚类，合并同义词
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np


class EntityClusterer:
    """
    Entity聚类器（per-relation）

    参考RelationClusterer的结构，但针对entity层：
    - 对每个relation独立处理
    - 使用BERTopic自适应聚类
    """

    def __init__(self, stage1_5_result_path: str):
        """
        初始化

        Args:
            stage1_5_result_path: Stage 1.5重分配结果JSON路径
        """
        self.stage1_5_result_path = Path(stage1_5_result_path)
        self.entities_by_relation = {}
        self.entity_counts_by_relation = {}

    def load_stage1_5_results(self) -> Dict:
        """
        加载Stage 1.5重分配结果

        Returns:
            Stage 1.5数据字典
        """
        with open(self.stage1_5_result_path, 'r') as f:
            data = json.load(f)

        print(f"✓ Loaded Stage 1.5 results from: {self.stage1_5_result_path}")
        print(f"  Total relations: {data['metadata']['total_relations']}")

        return data

    def extract_entities_by_relation(self, stage1_5_data: Dict) -> Dict[str, List[str]]:
        """
        按relation提取entities

        Args:
            stage1_5_data: Stage 1.5数据

        Returns:
            {relation: [entity1, entity2, ...]}（包含重复，用于频率统计）
        """
        redistributed = stage1_5_data.get('redistributed', {})

        for relation, entities in redistributed.items():
            entity_list = []
            for entity_data in entities:
                entity = entity_data['entity']
                entity_list.append(entity)

            self.entities_by_relation[relation] = entity_list
            self.entity_counts_by_relation[relation] = Counter(entity_list)

        print(f"\n✓ Extracted entities by relation:")
        for relation, entity_counter in self.entity_counts_by_relation.items():
            print(f"  {relation:25s}: {len(entity_counter):3d} unique entities, "
                  f"{sum(entity_counter.values()):4d} total instances")

        return self.entities_by_relation

    def embed_entities_bge(
        self,
        relation: str,
        model_name: str = "BAAI/bge-base-en-v1.5"
    ) -> np.ndarray:
        """
        使用BGE模型生成entity embeddings（单个relation）

        Args:
            relation: Relation名称
            model_name: BGE模型名称

        Returns:
            embeddings矩阵 [n_entities, embed_dim]
        """
        from sentence_transformers import SentenceTransformer

        if relation not in self.entity_counts_by_relation:
            raise ValueError(f"Relation '{relation}' not found")

        unique_entities = list(self.entity_counts_by_relation[relation].keys())

        print(f"\n🔄 Generating embeddings for {len(unique_entities)} entities in '{relation}'...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            unique_entities,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        print(f"✓ Embeddings shape: {embeddings.shape}")

        return embeddings, unique_entities

    def cluster_with_bertopic(
        self,
        relation: str,
        embeddings: np.ndarray,
        unique_entities: List[str],
        min_cluster_size: int = 2,
        verbose: bool = True
    ) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        使用BERTopic进行聚类（自适应聚类数）

        Args:
            relation: Relation名称
            embeddings: Entity embeddings
            unique_entities: Entity列表
            min_cluster_size: HDBSCAN最小cluster大小
            verbose: 是否输出详细信息

        Returns:
            (topic_model, topics, probabilities)
        """
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN

        print(f"\n{'='*60}")
        print(f"BERTopic聚类: {relation}")
        print(f"{'='*60}")

        # 自定义UMAP
        umap_model = UMAP(
            n_components=5,
            n_neighbors=min(15, len(unique_entities) - 1),
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # 自定义HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # 创建BERTopic模型（不使用embedding_model，直接用预计算的embeddings）
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=verbose,
            calculate_probabilities=True
        )

        # 聚类（使用预计算的embeddings）
        topics, probs = topic_model.fit_transform(unique_entities, embeddings)

        # 分析结果
        n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
        n_noise = sum(1 for t in topics if t == -1)

        print(f"\n✓ BERTopic聚类完成:")
        print(f"  发现的clusters: {n_clusters}")
        print(f"  噪声点数量: {n_noise}")
        print(f"  Entities: {len(unique_entities)} → {n_clusters} clusters")

        # 显示每个topic的entities
        entity_counter = self.entity_counts_by_relation[relation]
        print(f"\n  Topic分布:")

        for topic_id in sorted(set(topics)):
            topic_entities = [
                unique_entities[i]
                for i, t in enumerate(topics)
                if t == topic_id
            ]
            topic_counts = [entity_counter[ent] for ent in topic_entities]
            total_count = sum(topic_counts)

            if topic_id == -1:
                print(f"    Topic {topic_id:3d} (Noise): {len(topic_entities):2d} entities "
                      f"({total_count:3d} instances)")
            else:
                print(f"    Topic {topic_id:3d}: {len(topic_entities):2d} entities "
                      f"({total_count:3d} instances) - {', '.join(topic_entities[:3])}")

        return topic_model, topics, probs

    def generate_entity_mapping(
        self,
        relation: str,
        topics: np.ndarray,
        unique_entities: List[str],
        method: str = 'frequency'
    ) -> Dict[str, str]:
        """
        生成entity映射表（原始 → canonical）

        Args:
            relation: Relation名称
            topics: BERTopic的topic标签
            unique_entities: Entity列表
            method: 选择canonical name的方法（'frequency' 或 'alphabetical'）

        Returns:
            entity_mapping: {原始entity: canonical entity}
        """
        entity_mapping = {}
        canonical_entities = []
        entity_counter = self.entity_counts_by_relation[relation]

        # 按topic分组
        unique_topics = sorted(set(topics))

        for topic_id in unique_topics:
            # 获取该topic的所有entities
            topic_entities = [
                unique_entities[i]
                for i, t in enumerate(topics)
                if t == topic_id
            ]

            if topic_id == -1:
                # 噪声点：每个entity保持独立
                for ent in topic_entities:
                    entity_mapping[ent] = ent
                    canonical_entities.append(ent)
            else:
                # 正常topic：选择canonical name
                if method == 'frequency':
                    # 选择频次最高的作为canonical name
                    topic_counts = [
                        (ent, entity_counter[ent])
                        for ent in topic_entities
                    ]
                    topic_counts.sort(key=lambda x: x[1], reverse=True)
                    canonical_name = topic_counts[0][0]
                else:
                    # 按字母顺序选择第一个
                    canonical_name = sorted(topic_entities)[0]

                canonical_entities.append(canonical_name)

                # 建立映射
                for ent in topic_entities:
                    entity_mapping[ent] = canonical_name

        print(f"\n✓ Generated entity mapping for '{relation}':")
        print(f"  Original entities: {len(entity_mapping)}")
        print(f"  Canonical entities: {len(canonical_entities)}")
        print(f"  Compression: {len(canonical_entities)/len(entity_mapping)*100:.1f}%")

        return entity_mapping

    def save_results(
        self,
        all_entity_mappings: Dict[str, Dict[str, str]],
        output_path: str,
        method: str = 'bertopic',
        metadata: Optional[Dict] = None
    ):
        """
        保存聚类结果

        Args:
            all_entity_mappings: {relation: {original_entity: canonical_entity}}
            output_path: 输出路径
            method: 聚类方法名称
            metadata: 额外的元数据
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 统计
        total_entities_before = sum(
            len(counter) for counter in self.entity_counts_by_relation.values()
        )
        total_entities_after = sum(
            len(set(mapping.values())) for mapping in all_entity_mappings.values()
        )

        # 构建输出数据
        output_data = {
            'metadata': {
                'method': method,
                'source_file': str(self.stage1_5_result_path),
                'total_relations': len(all_entity_mappings),
                'total_entities_before': total_entities_before,
                'total_entities_after': total_entities_after,
                'compression_ratio': total_entities_after / total_entities_before,
                **(metadata or {})
            },
            'entity_mappings': all_entity_mappings,
            'entity_counts': {
                rel: dict(counter)
                for rel, counter in self.entity_counts_by_relation.items()
            }
        }

        # 保存
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n" + "="*80)
        print(f"✓ Results saved to: {output_path}")
        print(f"  Total entities: {total_entities_before} → {total_entities_after}")
        print(f"  Compression ratio: {total_entities_after/total_entities_before:.2%}")
        print("="*80)


if __name__ == '__main__':
    # 测试代码
    print("EntityClusterer模块已加载")
    print("使用示例见 notebooks/phase2b_entity_clustering.ipynb")
