"""
Entity Refiner - LLM后处理修正语义冲突

用LLM检查BERTopic聚类结果，修正语义不一致的clusters
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from openai import OpenAI


class EntityRefiner:
    """使用LLM后处理修正entity聚类中的语义冲突"""

    def __init__(self, mapping_file: str):
        self.mapping_file = Path(mapping_file)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.data = None
        self.entity_mappings = None

    def load_mappings(self):
        """加载BERTopic聚类结果"""
        with open(self.mapping_file, 'r') as f:
            self.data = json.load(f)

        self.entity_mappings = self.data['entity_mappings']
        print(f"✓ 加载了 {len(self.entity_mappings)} 个relations的mappings")

    def get_clusters(self, relation: str) -> Dict[str, List[str]]:
        """从mapping提取clusters"""
        mapping = self.entity_mappings[relation]
        clusters = {}

        for entity, canonical in mapping.items():
            if canonical not in clusters:
                clusters[canonical] = []
            clusters[canonical].append(entity)

        return clusters

    def check_cluster_coherence(self, relation: str, canonical: str, entities: List[str]) -> Dict:
        """用LLM检查cluster的语义一致性"""
        if len(entities) == 1:
            return {"coherent": True, "groups": [entities]}

        prompt = f"""Relation: {relation}
Cluster name: {canonical}
Entities: {', '.join(entities)}

Are these entities semantically coherent for this relation?

Rules:
- Opposite meanings (like "protagonist" vs "antagonist") should NOT be together
- Very different concepts should NOT be together
- Similar/related concepts CAN be together

Output ONLY valid JSON:
{{
  "coherent": true/false,
  "reason": "brief explanation",
  "groups": [["entity1", "entity2"], ["entity3"]]
}}

If coherent=true, put all in one group. If coherent=false, split into 2-3 coherent groups."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def refine_relation(self, relation: str) -> Dict[str, str]:
        """对单个relation进行LLM检查和修正"""
        print(f"\n{'='*80}")
        print(f"检查: {relation}")
        print(f"{'='*80}")

        clusters = self.get_clusters(relation)
        print(f"原始clusters: {len(clusters)}")

        new_mapping = {}
        issues_found = 0

        for canonical, entities in sorted(clusters.items()):
            # 跳过other_xxx
            if canonical.startswith('other_'):
                for ent in entities:
                    new_mapping[ent] = canonical
                continue

            # LLM检查
            result = self.check_cluster_coherence(relation, canonical, entities)

            if result['coherent']:
                # 保持原样
                for ent in entities:
                    new_mapping[ent] = canonical
            else:
                # 需要分离
                issues_found += 1
                print(f"\n⚠️  发现语义冲突: {canonical}")
                print(f"   原因: {result['reason']}")
                print(f"   分离成 {len(result['groups'])} 组")

                for group in result['groups']:
                    group_canonical = sorted(group)[0]  # 字母序第一个
                    for ent in group:
                        new_mapping[ent] = group_canonical
                    print(f"     → {group_canonical}: {group}")

        n_before = len(clusters)
        n_after = len(set(new_mapping.values()))

        if issues_found > 0:
            print(f"\n✓ 修正完成: {n_before} → {n_after} clusters (发现{issues_found}个冲突)")
        else:
            print(f"✓ 无冲突，保持原样: {n_before} clusters")

        return new_mapping

    def refine_all(self) -> Dict[str, Dict[str, str]]:
        """对所有relations进行检查修正"""
        refined_mappings = {}

        for relation in sorted(self.entity_mappings.keys()):
            refined_mappings[relation] = self.refine_relation(relation)

        return refined_mappings

    def save_results(self, refined_mappings: Dict, output_path: str):
        """保存修正后的结果"""
        output_path = Path(output_path)

        # 统计
        total_before = sum(len(set(self.entity_mappings[r].values())) for r in refined_mappings.keys())
        total_after = sum(len(set(m.values())) for m in refined_mappings.values())

        output_data = {
            'metadata': {
                **self.data['metadata'],
                'refinement': 'LLM post-processing',
                'total_clusters_before': total_before,
                'total_clusters_after': total_after,
                'note': 'Semantically conflicting clusters split by LLM'
            },
            'entity_mappings': refined_mappings
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ 结果已保存: {output_path}")
        print(f"   总clusters: {total_before} → {total_after}")
