#!/usr/bin/env python3
"""
构建Amazon Video Games的知识图谱（RecBole格式）

输入：
- results/phase3_full_extraction.json: Phase 3提取的知识点
- [交互数据文件]: 用户-物品交互数据（需要指定路径）

输出：
- data/recbole/amazon-videogames/amazon-videogames.item.kg: Item知识图谱
- data/recbole/amazon-videogames/amazon-videogames.user.kg: User兴趣图谱
- data/recbole/amazon-videogames/id_mappings.json: ID映射表
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AmazonVideoGamesKGBuilder:
    """Amazon Video Games知识图谱构建器"""

    def __init__(
        self,
        extraction_path: str,
        inter_path: str = None,
        output_dir: str = "data/recbole/amazon-videogames",
        min_rating: float = 4.0
    ):
        """
        Args:
            extraction_path: Phase 3提取结果路径
            inter_path: 交互数据路径（可选，用于构建user KG）
            output_dir: 输出目录
            min_rating: 最小评分阈值（用于过滤正样本）
        """
        self.extraction_path = Path(extraction_path)
        self.inter_path = Path(inter_path) if inter_path else None
        self.output_dir = Path(output_dir)
        self.min_rating = min_rating

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ID映射表
        self.item_id_map = {}  # asin -> recbole_id
        self.entity_id_map = {}  # entity_name -> entity_id

    def load_extraction_results(self) -> Dict:
        """加载Phase 3提取结果"""
        logger.info(f"Loading extraction results from {self.extraction_path}")

        with open(self.extraction_path, 'r') as f:
            data = json.load(f)

        results = data.get('results', [])
        logger.info(f"  Loaded {len(results)} items")

        return results

    def build_item_kg(self, results: List[Dict]):
        """构建Item知识图谱"""
        logger.info("\n=== Building Item Knowledge Graph ===")

        # 收集所有entity并构建ID映射
        all_entities = set()
        item_kps = {}  # item_id -> knowledge_points

        for item in results:
            recbole_id = item.get('recbole_id')
            asin = item.get('asin')
            kps = item.get('knowledge_points', [])

            if recbole_id is None:
                continue

            # 记录item ID映射
            self.item_id_map[asin] = recbole_id

            # 收集knowledge points
            item_kps[recbole_id] = kps

            # 收集entities
            for kp in kps:
                entity = kp.get('entity')
                if entity:
                    all_entities.add(entity)

        # 构建entity ID映射（按字母顺序）
        sorted_entities = sorted(all_entities)
        self.entity_id_map = {
            entity: idx + 1  # 从1开始（RecBole约定）
            for idx, entity in enumerate(sorted_entities)
        }

        logger.info(f"  Total items with KPs: {len(item_kps)}")
        logger.info(f"  Total unique entities: {len(self.entity_id_map)}")

        # 生成item.kg文件
        output_path = self.output_dir / "amazon-videogames.item.kg"

        triplets = []
        for item_id, kps in item_kps.items():
            for kp in kps:
                relation = kp.get('relation')
                entity = kp.get('entity')

                if relation and entity:
                    triplets.append({
                        'head_id': item_id,
                        'relation_id': relation,
                        'tail_id': entity
                    })

        # 保存为TSV
        df = pd.DataFrame(triplets)
        df.to_csv(
            output_path,
            sep='\t',
            index=False,
            header=['head_id:token', 'relation_id:token', 'tail_id:token']
        )

        logger.info(f"  Saved {len(triplets)} triplets to {output_path}")

        # 统计
        relation_counts = Counter(t['relation_id'] for t in triplets)
        logger.info(f"\n  Relation distribution:")
        for rel, count in relation_counts.most_common(10):
            logger.info(f"    {rel}: {count}")

        return item_kps

    def build_user_kg(self, item_kps: Dict):
        """构建User兴趣图谱（基于交互数据）"""
        if self.inter_path is None:
            logger.warning("\n⚠️  No interaction file provided, skipping user KG generation")
            return

        logger.info(f"\n=== Building User Interest Knowledge Graph ===")
        logger.info(f"  Reading interactions from {self.inter_path}")

        # 读取交互数据
        # 假设格式：user_id, item_id, rating, timestamp
        # 需要根据实际格式调整
        try:
            inter_df = pd.read_csv(self.inter_path)
            logger.info(f"  Loaded {len(inter_df)} interactions")
        except Exception as e:
            logger.error(f"  Failed to load interaction file: {e}")
            logger.info("  You may need to prepare the interaction file first")
            return

        # 过滤：只保留高评分交互（正样本）
        if 'rating' in inter_df.columns:
            inter_df = inter_df[inter_df['rating'] >= self.min_rating]
            logger.info(f"  Filtered to {len(inter_df)} positive interactions (rating >= {self.min_rating})")

        # 构建用户-知识点映射
        user_entities = defaultdict(lambda: defaultdict(int))  # user -> entity -> count

        for _, row in inter_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']  # 应该是recbole_id

            # 获取该item的知识点
            kps = item_kps.get(item_id, [])

            for kp in kps:
                entity = kp.get('entity')
                if entity:
                    user_entities[user_id][entity] += 1

        logger.info(f"  Collected entities for {len(user_entities)} users")

        # 为每个用户生成长期和短期兴趣
        user_kg_triplets = []

        for user_id, entity_counts in user_entities.items():
            # 计算TF-IDF风格的权重
            total_count = sum(entity_counts.values())

            # 长期兴趣：高频且独特的entity（TF-IDF高）
            # 简化版：使用频率 * log(独特性)

            # 短期兴趣：纯高频entity
            sorted_entities = sorted(
                entity_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Top 20% 作为短期兴趣
            num_short = max(1, int(len(sorted_entities) * 0.2))
            short_term = sorted_entities[:num_short]

            # 中等频率作为长期兴趣（20%-60%）
            num_long_start = num_short
            num_long_end = max(num_short + 1, int(len(sorted_entities) * 0.6))
            long_term = sorted_entities[num_long_start:num_long_end]

            # 添加短期兴趣
            for entity, count in short_term:
                user_kg_triplets.append({
                    'head_id': user_id,
                    'relation_id': 'short_term_interest',
                    'tail_id': entity
                })

            # 添加长期兴趣
            for entity, count in long_term:
                user_kg_triplets.append({
                    'head_id': user_id,
                    'relation_id': 'long_term_interest',
                    'tail_id': entity
                })

        # 保存user.kg
        output_path = self.output_dir / "amazon-videogames.user.kg"

        df = pd.DataFrame(user_kg_triplets)
        df.to_csv(
            output_path,
            sep='\t',
            index=False,
            header=['head_id:token', 'relation_id:token', 'tail_id:token']
        )

        logger.info(f"  Saved {len(user_kg_triplets)} user-entity edges to {output_path}")

        # 统计
        num_short = sum(1 for t in user_kg_triplets if t['relation_id'] == 'short_term_interest')
        num_long = sum(1 for t in user_kg_triplets if t['relation_id'] == 'long_term_interest')
        logger.info(f"    Short-term interests: {num_short}")
        logger.info(f"    Long-term interests: {num_long}")

    def save_id_mappings(self):
        """保存ID映射表"""
        mappings = {
            'item_id_map': self.item_id_map,
            'entity_id_map': self.entity_id_map
        }

        output_path = self.output_dir / "id_mappings.json"
        with open(output_path, 'w') as f:
            json.dump(mappings, f, indent=2)

        logger.info(f"\n✓ Saved ID mappings to {output_path}")

    def build(self):
        """完整构建流程"""
        logger.info("=" * 70)
        logger.info("Amazon Video Games Knowledge Graph Builder")
        logger.info("=" * 70)

        # 1. 加载提取结果
        results = self.load_extraction_results()

        # 2. 构建Item KG
        item_kps = self.build_item_kg(results)

        # 3. 构建User KG（如果有交互数据）
        self.build_user_kg(item_kps)

        # 4. 保存ID映射
        self.save_id_mappings()

        logger.info("\n" + "=" * 70)
        logger.info("✓ Knowledge Graph Construction Complete!")
        logger.info("=" * 70)
        logger.info(f"\nOutput files in {self.output_dir}:")
        logger.info(f"  - amazon-videogames.item.kg")
        if self.inter_path:
            logger.info(f"  - amazon-videogames.user.kg")
        logger.info(f"  - id_mappings.json")


def main():
    parser = argparse.ArgumentParser(description='Build Amazon Video Games Knowledge Graph')

    parser.add_argument(
        '--extraction',
        type=str,
        default='results/phase3_full_extraction.json',
        help='Path to Phase 3 extraction results'
    )

    parser.add_argument(
        '--inter',
        type=str,
        default=None,
        help='Path to interaction file (CSV with user_id, item_id, rating columns)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/recbole/amazon-videogames',
        help='Output directory'
    )

    parser.add_argument(
        '--min-rating',
        type=float,
        default=4.0,
        help='Minimum rating threshold for positive samples'
    )

    args = parser.parse_args()

    # 构建知识图谱
    builder = AmazonVideoGamesKGBuilder(
        extraction_path=args.extraction,
        inter_path=args.inter,
        output_dir=args.output_dir,
        min_rating=args.min_rating
    )

    builder.build()


if __name__ == '__main__':
    main()
