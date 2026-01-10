#!/usr/bin/env python3
"""
Phase 2b Stage 2: Entity Clustering (Per-Relation)
将Stage 1.5重分配后的entities按relation聚类，合并同义词

基于Phase 2a relation_clustering的成功经验改编
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import argparse
from src.clustering.entity_clusterer import EntityClusterer


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2b Stage 2: Entity Clustering (Per-Relation)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/entity_redistribution_stage1_5_redistributed.json',
        help='Stage 1.5重分配结果JSON路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/entity_clustering_stage2_merged.json',
        help='输出JSON路径'
    )
    parser.add_argument(
        '--relations',
        type=str,
        nargs='+',
        default=None,
        help='指定要处理的relations（默认处理全部）'
    )
    parser.add_argument(
        '--target-compression',
        type=float,
        default=0.33,
        help='目标压缩比（默认0.33，即压缩到约1/3）'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='BAAI/bge-base-en-v1.5',
        help='BGE模型名称'
    )
    parser.add_argument(
        '--linkage',
        type=str,
        default='average',
        choices=['average', 'ward', 'complete'],
        help='Agglomerative聚类的linkage方法'
    )

    args = parser.parse_args()

    print("="*80)
    print("Phase 2b Stage 2: Entity Clustering (Per-Relation)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Target compression ratio: {args.target_compression:.2%}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Linkage: {args.linkage}")

    # ========================================
    # 1. 初始化聚类器并加载数据
    # ========================================
    clusterer = EntityClusterer(
        stage1_5_result_path=args.input
    )

    stage1_5_data = clusterer.load_stage1_5_results()

    # ========================================
    # 2. 提取entities（按relation分组）
    # ========================================
    entity_counts_by_relation = clusterer.extract_entities_by_relation(stage1_5_data)

    # 确定要处理的relations
    if args.relations:
        relations_to_process = args.relations
    else:
        relations_to_process = sorted(entity_counts_by_relation.keys())

    print(f"\n{'='*80}")
    print(f"Processing {len(relations_to_process)} relations")
    print(f"{'='*80}")

    # ========================================
    # 3. 对每个relation进行聚类
    # ========================================
    all_entity_mappings = {}

    for idx, relation in enumerate(relations_to_process, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(relations_to_process)}] Processing relation: {relation}")
        print(f"{'='*80}")

        n_entities = len(entity_counts_by_relation[relation])

        # 如果entity数量太少（<=2），跳过聚类
        if n_entities <= 2:
            print(f"⚠️  Skipping '{relation}' (only {n_entities} entities, no clustering needed)")
            # 直接创建identity mapping
            all_entity_mappings[relation] = {
                ent: ent
                for ent in entity_counts_by_relation[relation].keys()
            }
            continue

        # 3a. 生成BGE embeddings
        embeddings, unique_entities = clusterer.embed_entities_bge(
            relation=relation,
            model_name=args.embedding_model
        )

        # 3b. Agglomerative聚类
        labels, clustering_model, n_clusters = clusterer.cluster_with_agglomerative(
            relation=relation,
            embeddings=embeddings,
            unique_entities=unique_entities,
            target_compression_ratio=args.target_compression,
            linkage=args.linkage
        )

        # 3c. 生成entity映射
        entity_mapping = clusterer.generate_entity_mapping(
            relation=relation,
            labels=labels,
            unique_entities=unique_entities,
            method='frequency'  # 选择频次最高的作为canonical name
        )

        all_entity_mappings[relation] = entity_mapping

    # ========================================
    # 4. 保存结果
    # ========================================
    clusterer.save_results(
        all_entity_mappings=all_entity_mappings,
        output_path=args.output,
        metadata={
            'target_compression_ratio': args.target_compression,
            'embedding_model': args.embedding_model,
            'linkage': args.linkage,
            'relations_processed': len(relations_to_process)
        }
    )

    # ========================================
    # 5. 最终统计
    # ========================================
    print(f"\n{'='*80}")
    print("Final Statistics")
    print(f"{'='*80}")

    for relation in sorted(all_entity_mappings.keys()):
        mapping = all_entity_mappings[relation]
        n_before = len(mapping)
        n_after = len(set(mapping.values()))
        compression = n_after / n_before if n_before > 0 else 1.0

        print(f"  {relation:25s}: {n_before:3d} → {n_after:3d} "
              f"(compression: {compression:.2%})")

    total_before = sum(len(m) for m in all_entity_mappings.values())
    total_after = sum(len(set(m.values())) for m in all_entity_mappings.values())

    print(f"\n  {'TOTAL':25s}: {total_before:3d} → {total_after:3d} "
          f"(compression: {total_after/total_before:.2%})")

    print(f"\n{'='*80}")
    print("✅ Phase 2b Stage 2 completed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
