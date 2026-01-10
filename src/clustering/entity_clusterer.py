"""
Entity聚类模块 - Phase 2b Stage 2
对Stage 1.5的final_entities按relation聚类，合并同义词
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class EntityClusterer:
    """Entity聚类器 - 使用BERTopic对每个relation的entities聚类"""

    def __init__(self, stage1_5_result_path: str):
        """
        初始化

        Args:
            stage1_5_result_path: Stage 1.5重分配结果JSON路径
        """
        self.stage1_5_result_path = Path(stage1_5_result_path)
        self.entities_by_relation = {}  # {relation: [entity1, entity2, ...]}

    def load_stage1_5_results(self) -> Dict:
        """加载Stage 1.5重分配结果，提取每个relation的final_entities"""
        with open(self.stage1_5_result_path, 'r') as f:
            data = json.load(f)

        print(f"✓ Loaded Stage 1.5 results from: {self.stage1_5_result_path}")

        # 提取每个relation的final_entities
        results = data.get('results', {})
        for relation, relation_data in results.items():
            final_entities = relation_data.get('final_entities', [])
            self.entities_by_relation[relation] = final_entities

        print(f"\n✓ Extracted entities from {len(self.entities_by_relation)} relations:")
        for relation in sorted(self.entities_by_relation.keys()):
            count = len(self.entities_by_relation[relation])
            print(f"  {relation:25s}: {count:3d} entities")

        return data

    def embed_entities_bge(
        self,
        relation: str,
        model_name: str = "BAAI/bge-base-en-v1.5"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        使用BGE模型生成entity embeddings

        Args:
            relation: Relation名称
            model_name: BGE模型名称

        Returns:
            (embeddings矩阵, entity列表)
        """
        from sentence_transformers import SentenceTransformer

        entities = self.entities_by_relation[relation]

        print(f"\n🔄 Generating embeddings for {len(entities)} entities in '{relation}'...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(entities, show_progress_bar=False, convert_to_numpy=True)

        print(f"✓ Embeddings shape: {embeddings.shape}")
        return embeddings, entities

    def cluster_with_bertopic(
        self,
        relation: str,
        embeddings: np.ndarray,
        entities: List[str],
        min_cluster_size: int = 2,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_dist: float = 0.0,
        verbose: bool = True
    ) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        使用BERTopic进行聚类

        Args:
            relation: Relation名称
            embeddings: Entity embeddings
            entities: Entity列表
            min_cluster_size: HDBSCAN最小cluster大小
            n_neighbors: UMAP邻居数
            n_components: UMAP降维维度
            min_dist: UMAP最小距离
            verbose: 是否输出详细信息

        Returns:
            (topic_model, topics, probabilities)
        """
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN

        print(f"\n{'='*60}")
        print(f"BERTopic聚类: {relation}")
        print(f"{'='*60}")
        if verbose:
            print(f"参数: min_cluster_size={min_cluster_size}, n_neighbors={n_neighbors}, "
                  f"n_components={n_components}, min_dist={min_dist}")

        # UMAP降维
        umap_model = UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, len(entities) - 1),
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )

        # HDBSCAN聚类
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # 创建BERTopic模型
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=verbose,
            calculate_probabilities=True
        )

        # 聚类
        topics, probs = topic_model.fit_transform(entities, embeddings)

        # 统计
        n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
        n_noise = sum(1 for t in topics if t == -1)

        print(f"\n✓ BERTopic聚类完成:")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise: {n_noise}")
        print(f"  {len(entities)} entities → {n_clusters} clusters")

        # 显示每个topic
        print(f"\n  Topic分布:")
        for topic_id in sorted(set(topics)):
            topic_entities = [entities[i] for i, t in enumerate(topics) if t == topic_id]

            if topic_id == -1:
                print(f"    Topic {topic_id:3d} (Noise): {len(topic_entities):2d} entities")
            else:
                print(f"    Topic {topic_id:3d}: {len(topic_entities):2d} entities - {', '.join(topic_entities[:3])}")

        return topic_model, topics, probs

    def generate_entity_mapping(
        self,
        relation: str,
        topics: np.ndarray,
        entities: List[str]
    ) -> Dict[str, str]:
        """
        生成entity映射表（按字母顺序选择canonical name）

        注意：所有noise entities归为一个"other_{relation}" canonical entity

        Args:
            relation: Relation名称
            topics: BERTopic的topic标签
            entities: Entity列表

        Returns:
            entity_mapping: {原始entity: canonical entity}
        """
        entity_mapping = {}
        canonical_entities = []

        for topic_id in sorted(set(topics)):
            topic_entities = [entities[i] for i, t in enumerate(topics) if t == topic_id]

            if topic_id == -1:
                # Noise: 所有noise归为一个"other_xxx"
                other_name = f"other_{relation}"
                for ent in topic_entities:
                    entity_mapping[ent] = other_name
                canonical_entities.append(other_name)
            else:
                # 正常topic: 按字母顺序选第一个作为canonical name
                canonical_name = sorted(topic_entities)[0]
                canonical_entities.append(canonical_name)

                for ent in topic_entities:
                    entity_mapping[ent] = canonical_name

        print(f"\n✓ Generated entity mapping for '{relation}':")
        print(f"  {len(entity_mapping)} entities → {len(canonical_entities)} canonical")
        print(f"  Compression: {len(canonical_entities)/len(entity_mapping)*100:.1f}%")

        return entity_mapping

    def save_results(
        self,
        all_entity_mappings: Dict[str, Dict[str, str]],
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """保存聚类结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 统计
        total_before = sum(len(self.entities_by_relation[rel]) for rel in all_entity_mappings.keys())
        total_after = sum(len(set(mapping.values())) for mapping in all_entity_mappings.values())

        # 构建输出数据
        output_data = {
            'metadata': {
                'method': 'bertopic',
                'source_file': str(self.stage1_5_result_path),
                'total_relations': len(all_entity_mappings),
                'total_entities_before': total_before,
                'total_entities_after': total_after,
                'compression_ratio': total_after / total_before,
                **(metadata or {})
            },
            'entity_mappings': all_entity_mappings
        }

        # 保存
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n" + "="*80)
        print(f"✓ Results saved to: {output_path}")
        print(f"  {total_before} entities → {total_after} canonical entities")
        print(f"  Compression: {total_after/total_before:.2%}")
        print("="*80)


if __name__ == '__main__':
    print("EntityClusterer模块已加载")
    print("使用示例见 notebooks/phase2b_entity_clustering.ipynb")
