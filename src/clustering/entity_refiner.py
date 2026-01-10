"""
Entity聚类微调模块 - 使用LLM修正BERTopic聚类的语义冲突

用于检测并分离语义不一致的entity clusters（如protagonist + antagonist）
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class EntityRefiner:
    """
    使用LLM检查entity聚类的语义一致性，修正BERTopic产生的语义冲突

    核心思想：LLM检查每个cluster的语义coherence，对不一致的cluster进行split
    """

    def __init__(self, mapping_file: str):
        """
        初始化

        Args:
            mapping_file: entity mapping文件路径（BERTopic聚类结果）
        """
        self.mapping_file = Path(mapping_file)
        self.data = None
        self.entity_mappings = None

    def load_mappings(self):
        """加载BERTopic聚类结果"""
        with open(self.mapping_file, 'r') as f:
            self.data = json.load(f)

        self.entity_mappings = self.data['entity_mappings']
        print(f"✓ 加载了 {len(self.entity_mappings)} 个relations的mappings")

        # 统计总clusters
        total_clusters = sum(len(set(m.values())) for m in self.entity_mappings.values())
        print(f"  总clusters: {total_clusters}")

    def get_clusters(self, relation: str) -> Dict[str, List[str]]:
        """
        从mapping提取clusters

        Args:
            relation: relation名称

        Returns:
            {canonical_entity: [entity1, entity2, ...]}
        """
        mapping = self.entity_mappings[relation]
        clusters = defaultdict(list)

        for entity, canonical in mapping.items():
            clusters[canonical].append(entity)

        return dict(clusters)

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> str:
        """调用OpenAI API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        # 初始化客户端
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url

        client = OpenAI(**client_kwargs)

        # 调用API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    def check_cluster_coherence(
        self,
        relation: str,
        canonical: str,
        entities: List[str],
        backend: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> Dict:
        """
        用LLM检查cluster的语义一致性

        Args:
            relation: relation名称
            canonical: canonical entity名称
            entities: cluster中的entities列表
            其他: LLM配置参数

        Returns:
            {"coherent": bool, "reason": str, "groups": [[...], [...]]}
        """
        if len(entities) == 1:
            return {"coherent": True, "groups": [entities]}

        system_prompt = """You are an expert in semantic analysis for knowledge graph construction.

Your task is to check if entities grouped together in a cluster are semantically coherent for the given relation.

RULES:
1. Opposite meanings (like "protagonist" vs "antagonist") should NOT be together
2. Very different concepts (like "weapons" vs "beverage") should NOT be together
3. Similar or related concepts CAN be together
4. Consider the context of movie poster knowledge extraction

IMPORTANT: Focus on semantic meaning, not just word similarity."""

        user_prompt = f"""Relation: {relation}
Cluster name: {canonical}
Entities in this cluster: {', '.join(entities)}

Are these entities semantically coherent for this relation?

**Output Format:**
Return a JSON object:
```json
{{
  "coherent": true or false,
  "reason": "brief explanation of why they are/aren't coherent",
  "groups": [
    ["entity1", "entity2"],
    ["entity3"]
  ]
}}
```

**Guidelines:**
- If coherent=true: put all entities in one group
- If coherent=false: split into 2-3 semantically coherent sub-groups
- Each entity must appear in exactly one group
- Each group should represent a distinct semantic concept

Now analyze this cluster:"""

        if backend == 'openai':
            response = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature
            )
        else:
            raise ValueError(f"不支持的backend: {backend}")

        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始响应:\n{response}")
            raise

        return result

    def refine_relation(
        self,
        relation: str,
        backend: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> Dict[str, str]:
        """
        对单个relation进行LLM检查和修正

        Args:
            relation: relation名称
            其他: LLM配置参数

        Returns:
            修正后的entity mapping {entity: canonical}
        """
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
            result = self.check_cluster_coherence(
                relation=relation,
                canonical=canonical,
                entities=entities,
                backend=backend,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature
            )

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
                    # 使用字母序第一个作为新的canonical
                    group_canonical = sorted(group)[0]
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

    def refine_all(
        self,
        backend: str = 'openai',
        model_name: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Dict[str, str]]:
        """
        对所有relations进行检查修正

        Args:
            backend: LLM后端
            model_name: 模型名称
            api_key: API密钥
            base_url: API base URL
            temperature: 温度参数

        Returns:
            精炼后的mappings {relation: {entity: canonical}}
        """
        print(f"\n{'='*60}")
        print(f"使用LLM检查Entity聚类语义一致性")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")

        refined_mappings = {}

        for relation in sorted(self.entity_mappings.keys()):
            refined_mappings[relation] = self.refine_relation(
                relation=relation,
                backend=backend,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature
            )

        return refined_mappings

    def save_results(
        self,
        refined_mappings: Dict[str, Dict[str, str]],
        output_path: str
    ):
        """
        保存修正后的结果

        Args:
            refined_mappings: 精炼后的mappings
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 统计
        total_before = sum(len(set(self.entity_mappings[r].values())) for r in refined_mappings.keys())
        total_after = sum(len(set(m.values())) for m in refined_mappings.values())

        # 统计有多少relations发生了变化
        changed_relations = []
        for relation in refined_mappings.keys():
            before = len(set(self.entity_mappings[relation].values()))
            after = len(set(refined_mappings[relation].values()))
            if before != after:
                changed_relations.append({
                    'relation': relation,
                    'clusters_before': before,
                    'clusters_after': after
                })

        output_data = {
            'metadata': {
                **self.data['metadata'],
                'refinement_method': 'llm_semantic_validation',
                'llm_model': 'gpt-4o-mini',
                'total_clusters_before': total_before,
                'total_clusters_after': total_after,
                'relations_modified': len(changed_relations),
                'changed_relations': changed_relations,
                'note': 'Semantically conflicting clusters split by LLM'
            },
            'entity_mappings': refined_mappings
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n{'='*80}")
        print("✅ 结果已保存")
        print(f"{'='*80}")
        print(f"输出文件: {output_path}")
        print(f"总clusters: {total_before} → {total_after}")
        print(f"修改的relations: {len(changed_relations)}/{len(refined_mappings)}")

        if changed_relations:
            print(f"\n修改详情:")
            for item in changed_relations:
                print(f"  {item['relation']}: {item['clusters_before']} → {item['clusters_after']} clusters")


if __name__ == '__main__':
    print("EntityRefiner模块已加载")
    print("使用示例见 scripts/refine_entity_mapping.py")
