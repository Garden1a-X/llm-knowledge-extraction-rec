#!/usr/bin/env python3
"""
构建Amazon Beauty的知识图谱（RecBole格式）

输入：
- results/beauty/full_extraction.json: 全量提取的知识点
- data/beauty_entity_vocabulary.json: 词表（用于过滤invalid entities）

输出：
- data/recbole/amazon-beauty/amazon-beauty.kg: Item知识图谱
- data/recbole/amazon-beauty/kg_entity_map.json: Entity ID映射

Usage:
    python scripts/build_beauty_kg.py \
        --extraction results/beauty/full_extraction.json \
        --vocabulary data/beauty_entity_vocabulary.json \
        --output_dir data/recbole/amazon-beauty
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import argparse
import logging
import csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BeautyKGBuilder:
    """Amazon Beauty知识图谱构建器"""

    def __init__(
        self,
        extraction_path: str,
        vocabulary_path: str,
        output_dir: str = "data/recbole/amazon-beauty",
        include_new: bool = True,
        filter_invalid: bool = True
    ):
        """
        Args:
            extraction_path: 全量提取结果路径
            vocabulary_path: 词表路径
            output_dir: 输出目录
            include_new: 是否包含NEW_实体
            filter_invalid: 是否过滤invalid实体
        """
        self.extraction_path = Path(extraction_path)
        self.vocabulary_path = Path(vocabulary_path)
        self.output_dir = Path(output_dir)
        self.include_new = include_new
        self.filter_invalid = filter_invalid

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载词表
        self.vocabulary = self._load_vocabulary()

        # ID映射
        self.entity_id_map = {}  # entity_name -> entity_id
        self.relation_id_map = {}  # relation_name -> relation_id

    def _load_vocabulary(self) -> Dict:
        """加载词表"""
        with open(self.vocabulary_path, 'r') as f:
            return json.load(f)

    def _is_valid_kp(self, relation: str, entity: str) -> Tuple[bool, str]:
        """
        检查knowledge point是否有效

        Returns:
            (is_valid, reason)
        """
        # 检查relation
        if relation not in self.vocabulary['relations']:
            return False, 'invalid_relation'

        # 检查entity
        if entity.startswith('NEW_'):
            if self.include_new:
                return True, 'new_entity'
            else:
                return False, 'new_entity_excluded'

        # 检查是否在词表中
        valid_entities = set(self.vocabulary['relations'][relation]['standard_entities'])
        if entity in valid_entities:
            return True, 'valid'
        else:
            if self.filter_invalid:
                return False, 'invalid_entity'
            else:
                return True, 'invalid_kept'

    def load_extraction_results(self) -> List[Dict]:
        """加载提取结果"""
        logger.info(f"Loading extraction results from {self.extraction_path}")

        with open(self.extraction_path, 'r') as f:
            data = json.load(f)

        results = data.get('results', [])
        success_results = [r for r in results if r.get('status') == 'success']

        logger.info(f"  Total items: {len(results)}")
        logger.info(f"  Successful: {len(success_results)}")

        return success_results

    def build_kg(self, results: List[Dict]) -> List[Dict]:
        """构建知识图谱"""
        logger.info("\n=== Building Knowledge Graph ===")

        triplets = []
        stats = {
            'total_kps': 0,
            'valid': 0,
            'new_entity': 0,
            'invalid_relation': 0,
            'invalid_entity': 0
        }

        all_entities = set()
        all_relations = set()

        for item in results:
            recbole_id = item.get('recbole_id')
            kps = item.get('knowledge_points', [])

            if recbole_id is None:
                continue

            for kp in kps:
                relation = kp.get('relation', '')
                entity = kp.get('entity', '')
                stats['total_kps'] += 1

                # 检查有效性
                is_valid, reason = self._is_valid_kp(relation, entity)

                if reason in stats:
                    stats[reason] += 1

                if is_valid:
                    triplets.append({
                        'item_id': recbole_id,
                        'relation': relation,
                        'entity': entity
                    })
                    all_entities.add(entity)
                    all_relations.add(relation)

        # 构建ID映射
        self.relation_id_map = {
            rel: idx for idx, rel in enumerate(sorted(all_relations))
        }
        self.entity_id_map = {
            ent: idx for idx, ent in enumerate(sorted(all_entities))
        }

        logger.info(f"\n  Statistics:")
        logger.info(f"    Total KPs: {stats['total_kps']}")
        logger.info(f"    Valid: {stats['valid']}")
        logger.info(f"    NEW entities: {stats['new_entity']}")
        logger.info(f"    Invalid relations: {stats['invalid_relation']}")
        logger.info(f"    Invalid entities: {stats['invalid_entity']}")

        logger.info(f"\n  KG Statistics:")
        logger.info(f"    Total triplets: {len(triplets)}")
        logger.info(f"    Unique entities: {len(all_entities)}")
        logger.info(f"    Unique relations: {len(all_relations)}")

        # 关系分布
        if len(triplets) > 0:
            relation_counts = Counter(t['relation'] for t in triplets)
            logger.info(f"\n  Relation distribution:")
            for rel, count in relation_counts.most_common():
                logger.info(f"    {rel}: {count}")

        return triplets

    def save_kg(self, triplets: List[Dict]):
        """保存Item知识图谱"""
        # 保存.item.kg文件（RecBole格式）
        kg_path = self.output_dir / "amazon-beauty.item.kg"

        # RecBole格式: head_id:token, relation_id:token, tail_id:token
        with open(kg_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['head_id:token', 'relation_id:token', 'tail_id:token'])
            for t in triplets:
                writer.writerow([t['item_id'], t['relation'], t['entity']])

        logger.info(f"\n✓ Saved KG to {kg_path}")
        logger.info(f"  {len(triplets)} triplets")

        # 保存entity映射
        entity_map_path = self.output_dir / "kg_entity_map.json"
        with open(entity_map_path, 'w') as f:
            json.dump({
                'entity_to_id': self.entity_id_map,
                'relation_to_id': self.relation_id_map,
                'num_entities': len(self.entity_id_map),
                'num_relations': len(self.relation_id_map)
            }, f, indent=2)
        logger.info(f"✓ Saved entity map to {entity_map_path}")

    def analyze_kg(self, triplets: List[Dict]):
        """分析知识图谱统计信息"""
        logger.info("\n=== KG Analysis ===")

        # 每个item的triplet数量分布
        item_counts = Counter(t['item_id'] for t in triplets)
        counts = list(item_counts.values())
        if counts:
            mean_count = sum(counts) / len(counts)
            sorted_counts = sorted(counts)
            median_count = sorted_counts[len(sorted_counts) // 2]
            logger.info(f"\n  Triplets per item:")
            logger.info(f"    Mean: {mean_count:.2f}")
            logger.info(f"    Median: {median_count}")
            logger.info(f"    Min: {min(counts)}")
            logger.info(f"    Max: {max(counts)}")

        # Entity频率分布
        entity_counts = Counter(t['entity'] for t in triplets)
        logger.info(f"\n  Top 20 entities:")
        for entity, count in entity_counts.most_common(20):
            logger.info(f"    {entity}: {count}")

        # NEW_ entities统计
        new_entity_counts = Counter(
            t['entity'] for t in triplets if t['entity'].startswith('NEW_')
        )
        logger.info(f"\n  NEW_ entities: {len(new_entity_counts)}")
        if new_entity_counts:
            logger.info(f"  Top 10 NEW_ entities:")
            for entity, count in new_entity_counts.most_common(10):
                logger.info(f"    {entity}: {count}")

    def build(self):
        """完整构建流程"""
        logger.info("=" * 70)
        logger.info("Amazon Beauty Knowledge Graph Builder")
        logger.info("=" * 70)
        logger.info(f"\nConfiguration:")
        logger.info(f"  Include NEW entities: {self.include_new}")
        logger.info(f"  Filter invalid entities: {self.filter_invalid}")

        # 1. 加载提取结果
        results = self.load_extraction_results()

        # 2. 构建KG
        df = self.build_kg(results)

        # 3. 分析KG
        self.analyze_kg(df)

        # 4. 保存KG
        self.save_kg(df)

        logger.info("\n" + "=" * 70)
        logger.info("✓ Knowledge Graph Construction Complete!")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Build Amazon Beauty Knowledge Graph')

    parser.add_argument(
        '--extraction',
        type=str,
        default='results/beauty/full_extraction.json',
        help='Path to full extraction results'
    )

    parser.add_argument(
        '--vocabulary',
        type=str,
        default='data/beauty_entity_vocabulary.json',
        help='Path to vocabulary file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/recbole/amazon-beauty',
        help='Output directory'
    )

    parser.add_argument(
        '--include_new',
        action='store_true',
        default=True,
        help='Include NEW_ entities in KG (default: True)'
    )

    parser.add_argument(
        '--no_include_new',
        action='store_true',
        help='Exclude NEW_ entities from KG'
    )

    parser.add_argument(
        '--keep_invalid',
        action='store_true',
        help='Keep invalid entities (not recommended)'
    )

    args = parser.parse_args()

    # 处理参数
    include_new = not args.no_include_new
    filter_invalid = not args.keep_invalid

    # 构建知识图谱
    builder = BeautyKGBuilder(
        extraction_path=args.extraction,
        vocabulary_path=args.vocabulary,
        output_dir=args.output_dir,
        include_new=include_new,
        filter_invalid=filter_invalid
    )

    builder.build()


if __name__ == '__main__':
    main()
