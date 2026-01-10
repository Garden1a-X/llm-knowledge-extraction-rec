"""
Entity聚类模块 - Phase 2b Stage 2
将Stage 1.5重分配后的entities（按relation）聚类合并同义词
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np


class EntityClusterer:
    """
    Entity聚类器（per-relation）

    基于Phase 2a RelationClusterer的成功经验，改编用于Entity层
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

    def extract_entities_by_relation(self, stage1_5_data: Dict) -> Dict[str, Counter]:
        """
        按relation提取entities

        Args:
            stage1_5_data: Stage 1.5数据

        Returns:
            {relation: Counter({entity: count})}
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

        return self.entity_counts_by_relation

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

        print(f"\n🔄 Generating embeddings for {len(unique_entities)} entities "
              f"in relation '{relation}'...")

        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            unique_entities,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        print(f"✓ Embeddings shape: {embeddings.shape}")

        return embeddings, unique_entities

    def cluster_with_agglomerative(
        self,
        relation: str,
        embeddings: np.ndarray,
        unique_entities: List[str],
        target_compression_ratio: float = 0.33,
        linkage: str = 'average'
    ) -> Tuple[np.ndarray, object, int]:
        """
        使用Agglomerative Clustering进行聚类（动态n_clusters）

        Args:
            relation: Relation名称
            embeddings: Entity embeddings
            unique_entities: Entity列表
            target_compression_ratio: 目标压缩比（约1/3）
            linkage: 连接方法 ('average', 'ward', 'complete')

        Returns:
            (cluster_labels, clustering_model, n_clusters)
        """
        from sklearn.cluster import AgglomerativeClustering

        n_entities = len(unique_entities)

        # 动态确定n_clusters（目标：压缩到约1/3）
        n_clusters = max(2, int(n_entities * target_compression_ratio))
        n_clusters = min(n_clusters, n_entities)  # 不能超过entity数

        print(f"\n🔄 Agglomerative Clustering for '{relation}':")
        print(f"  Input: {n_entities} entities")
        print(f"  Target n_clusters: {n_clusters} (compression ratio: {n_clusters/n_entities:.2%})")

        # Agglomerative聚类
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric='cosine'
        )

        labels = clustering.fit_predict(embeddings)

        # 分析结果
        entity_counter = self.entity_counts_by_relation[relation]

        print(f"\n✓ Clustering completed:")
        print(f"  Clusters: {n_clusters}")
        print(f"\n  Cluster distribution:")

        for cluster_id in range(n_clusters):
            cluster_entities = [
                unique_entities[i]
                for i, label in enumerate(labels)
                if label == cluster_id
            ]
            cluster_counts = [
                entity_counter[ent]
                for ent in cluster_entities
            ]
            total_count = sum(cluster_counts)

            print(f"    Cluster {cluster_id:2d}: {len(cluster_entities):2d} entities "
                  f"({total_count:4d} instances) - {', '.join(cluster_entities[:3])}")

        return labels, clustering, n_clusters

    def generate_entity_mapping(
        self,
        relation: str,
        labels: np.ndarray,
        unique_entities: List[str],
        method: str = 'frequency'
    ) -> Dict[str, str]:
        """
        生成entity映射表（原始 → canonical）

        Args:
            relation: Relation名称
            labels: Cluster标签
            unique_entities: Entity列表
            method: 选择canonical name的方法（'frequency' 或 'alphabetical'）

        Returns:
            entity_mapping: {原始entity: canonical entity}
        """
        entity_mapping = {}
        canonical_entities = []
        entity_counter = self.entity_counts_by_relation[relation]

        # 按cluster分组
        n_clusters = len(set(labels))

        for cluster_id in range(n_clusters):
            # 获取该cluster的所有entities
            cluster_entities = [
                unique_entities[i]
                for i, label in enumerate(labels)
                if label == cluster_id
            ]

            if method == 'frequency':
                # 选择频次最高的作为canonical name
                cluster_counts = [
                    (ent, entity_counter[ent])
                    for ent in cluster_entities
                ]
                cluster_counts.sort(key=lambda x: x[1], reverse=True)
                canonical_name = cluster_counts[0][0]
            else:
                # 按字母顺序选择第一个
                canonical_name = sorted(cluster_entities)[0]

            canonical_entities.append(canonical_name)

            # 建立映射
            for ent in cluster_entities:
                entity_mapping[ent] = canonical_name

        print(f"\n✓ Generated entity mapping for '{relation}':")
        print(f"  Canonical entities: {len(canonical_entities)}")

        return entity_mapping

    def save_results(
        self,
        all_entity_mappings: Dict[str, Dict[str, str]],
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        保存聚类结果

        Args:
            all_entity_mappings: {relation: {original_entity: canonical_entity}}
            output_path: 输出路径
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
                'method': 'agglomerative_per_relation',
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
    print("使用示例见 scripts/phase2b_stage2_entity_clustering.py")
