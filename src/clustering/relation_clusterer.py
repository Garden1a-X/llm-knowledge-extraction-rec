"""
Relation聚类模块 - Phase 2
将Phase 1提取的128个relation聚类到15-20个标准relation
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd


class RelationClusterer:
    """
    Relation聚类器

    支持两种聚类方法：
    1. BERTopic（自动探索，可视化）
    2. 传统方法（BGE + Agglomerative，精确控制）
    """

    def __init__(self, phase1_result_path: str):
        """
        初始化

        Args:
            phase1_result_path: Phase 1提取结果JSON路径
        """
        self.phase1_result_path = Path(phase1_result_path)
        self.relations = []
        self.relation_counts = None
        self.embeddings = None

    def load_phase1_results(self) -> Dict:
        """
        加载Phase 1提取结果

        Returns:
            Phase 1数据字典
        """
        with open(self.phase1_result_path, 'r') as f:
            data = json.load(f)

        print(f"✓ Loaded Phase 1 results from: {self.phase1_result_path}")
        print(f"  Total movies: {len(data.get('results', []))}")

        return data

    def extract_relations(self, phase1_data: Dict) -> Tuple[List[str], Counter]:
        """
        从Phase 1结果中提取所有relation

        Args:
            phase1_data: Phase 1数据

        Returns:
            (relation列表, relation计数器)
        """
        all_relations = []

        for result in phase1_data.get('results', []):
            if result.get('status') != 'success':
                continue

            for kp in result.get('knowledge_points', []):
                relation = kp.get('relation')
                if relation:
                    all_relations.append(relation)

        self.relations = all_relations
        self.relation_counts = Counter(all_relations)

        print(f"\n✓ Extracted relations:")
        print(f"  Total relation instances: {len(all_relations)}")
        print(f"  Unique relations: {len(self.relation_counts)}")
        print(f"\n  Top 10 relations:")
        for rel, count in self.relation_counts.most_common(10):
            print(f"    {rel:25s}: {count:4d}")

        return all_relations, self.relation_counts

    def embed_relations_bge(self, model_name: str = "BAAI/bge-base-en-v1.5") -> np.ndarray:
        """
        使用BGE模型生成relation embeddings

        Args:
            model_name: BGE模型名称

        Returns:
            embeddings矩阵 [n_relations, embed_dim]
        """
        from sentence_transformers import SentenceTransformer

        print(f"\n🔄 Loading BGE model: {model_name}...")
        model = SentenceTransformer(model_name)

        # 获取唯一relation列表
        unique_relations = list(self.relation_counts.keys())

        print(f"🔄 Generating embeddings for {len(unique_relations)} unique relations...")
        embeddings = model.encode(
            unique_relations,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.embeddings = embeddings
        self.unique_relations = unique_relations

        print(f"✓ Embeddings shape: {embeddings.shape}")

        return embeddings

    def cluster_with_bertopic(
        self,
        min_cluster_size: int = 3,
        n_components_umap: int = 10,
        verbose: bool = True
    ) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        使用BERTopic进行聚类

        Args:
            min_cluster_size: HDBSCAN最小cluster大小
            n_components_umap: UMAP降维维度
            verbose: 是否输出详细信息

        Returns:
            (topic_model, topics, probabilities)
        """
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN

        print(f"\n{'='*60}")
        print("方法1: BERTopic聚类")
        print(f"{'='*60}")

        # 自定义embedding模型
        embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

        # 自定义UMAP
        umap_model = UMAP(
            n_components=n_components_umap,
            n_neighbors=5,
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

        # 创建BERTopic模型
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=verbose,
            calculate_probabilities=True
        )

        # 聚类
        unique_relations = list(self.relation_counts.keys())
        topics, probs = topic_model.fit_transform(unique_relations)

        # 分析结果
        n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
        n_noise = sum(1 for t in topics if t == -1)

        print(f"\n✓ BERTopic聚类完成:")
        print(f"  发现的clusters: {n_clusters}")
        print(f"  噪声点数量: {n_noise}")

        # 显示每个topic的top relations
        print(f"\n  Topic分布:")
        topic_info = topic_model.get_topic_info()
        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            if topic_id == -1:
                print(f"    Topic {topic_id:3d} (Noise): {count} relations")
            else:
                # 获取该topic的relations
                topic_relations = [
                    unique_relations[i]
                    for i, t in enumerate(topics)
                    if t == topic_id
                ]
                print(f"    Topic {topic_id:3d}: {count} relations - {', '.join(topic_relations[:3])}")

        return topic_model, topics, probs

    def cluster_with_agglomerative(
        self,
        n_clusters: int = 18,
        linkage: str = 'average'
    ) -> Tuple[np.ndarray, object]:
        """
        使用Agglomerative Clustering进行聚类

        Args:
            n_clusters: 目标cluster数量
            linkage: 连接方法 ('average', 'ward', 'complete')

        Returns:
            (cluster_labels, clustering_model)
        """
        from sklearn.cluster import AgglomerativeClustering

        print(f"\n{'='*60}")
        print("方法2: Agglomerative Clustering")
        print(f"{'='*60}")

        if self.embeddings is None:
            raise ValueError("请先运行 embed_relations_bge() 生成embeddings")

        # Agglomerative聚类
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric='cosine'
        )

        labels = clustering.fit_predict(self.embeddings)

        # 分析结果
        print(f"\n✓ Agglomerative聚类完成:")
        print(f"  Clusters数量: {n_clusters}")
        print(f"  Linkage方法: {linkage}")

        # 显示每个cluster的成员
        print(f"\n  Cluster分布:")
        for cluster_id in range(n_clusters):
            cluster_relations = [
                self.unique_relations[i]
                for i, label in enumerate(labels)
                if label == cluster_id
            ]
            cluster_counts = [
                self.relation_counts[rel]
                for rel in cluster_relations
            ]
            total_count = sum(cluster_counts)

            print(f"    Cluster {cluster_id:2d}: {len(cluster_relations):2d} relations "
                  f"({total_count:4d} instances) - {', '.join(cluster_relations[:3])}")

        return labels, clustering

    def generate_relation_mapping(
        self,
        labels: np.ndarray,
        method: str = 'frequency'
    ) -> Dict[str, str]:
        """
        生成relation映射表（原始 → 标准）

        Args:
            labels: cluster标签
            method: 选择标准名称的方法（'frequency' 或 'manual'）

        Returns:
            relation_mapping: {原始relation: 标准relation}
        """
        relation_mapping = {}
        standard_relations = []

        # 按cluster分组
        n_clusters = len(set(labels))
        for cluster_id in range(n_clusters):
            # 获取该cluster的所有relations
            cluster_relations = [
                self.unique_relations[i]
                for i, label in enumerate(labels)
                if label == cluster_id
            ]

            if method == 'frequency':
                # 选择频次最高的作为标准名
                cluster_counts = [
                    (rel, self.relation_counts[rel])
                    for rel in cluster_relations
                ]
                cluster_counts.sort(key=lambda x: x[1], reverse=True)
                standard_name = cluster_counts[0][0]
            else:
                # 手动选择（默认第一个）
                standard_name = cluster_relations[0]

            standard_relations.append(standard_name)

            # 建立映射
            for rel in cluster_relations:
                relation_mapping[rel] = standard_name

        print(f"\n✓ 生成relation映射:")
        print(f"  标准relation数: {len(standard_relations)}")
        print(f"\n  标准relations:")
        for std_rel in standard_relations:
            count = self.relation_counts[std_rel]
            print(f"    - {std_rel:30s} ({count} instances)")

        return relation_mapping

    def save_results(
        self,
        relation_mapping: Dict[str, str],
        output_path: str,
        method: str,
        metadata: Optional[Dict] = None
    ):
        """
        保存聚类结果

        Args:
            relation_mapping: relation映射表
            output_path: 输出路径
            method: 聚类方法名称
            metadata: 额外的元数据
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 统计标准relations
        standard_relations = sorted(set(relation_mapping.values()))

        # 构建输出数据
        output_data = {
            'metadata': {
                'method': method,
                'source_file': str(self.phase1_result_path),
                'total_relation_instances': len(self.relations),
                'unique_relations_before': len(self.relation_counts),
                'standard_relations_after': len(standard_relations),
                **(metadata or {})
            },
            'standard_relations': standard_relations,
            'relation_mapping': relation_mapping,
            'relation_counts': dict(self.relation_counts)
        }

        # 保存
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 结果已保存到: {output_path}")


if __name__ == '__main__':
    # 测试代码
    print("RelationClusterer模块已加载")
    print("使用示例见 notebooks/phase2_relation_clustering.ipynb")
