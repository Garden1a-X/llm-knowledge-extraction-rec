"""
Entity聚类微调模块 - 使用LLM多轮优化entity聚类

三步法：
1. Split - 检查并分割语义冲突的clusters
2. Rename - 为每个cluster选择最佳的canonical name
3. Merge - 根据名字合并语义相似的clusters
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter


class EntityRefiner:
    """
    使用LLM多轮优化entity聚类

    工作流程：
    1. Split: 检查每个cluster，分割语义冲突的entities
    2. Rename: 为每个cluster选择最佳canonical name（不用字母序）
    3. Merge: 根据名字合并语义相似的clusters
    """

    def __init__(self, mapping_file: str, output_dir: str = 'results'):
        """
        初始化

        Args:
            mapping_file: entity mapping文件路径（BERTopic聚类结果）
            output_dir: 输出目录
        """
        self.mapping_file = Path(mapping_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = None
        self.entity_mappings = None
        self.current_mappings = None

    def load_mappings(self, from_file: Optional[str] = None):
        """
        加载entity mappings

        Args:
            from_file: 如果指定，从该文件加载（用于继续上次的结果）
        """
        load_path = Path(from_file) if from_file else self.mapping_file

        with open(load_path, 'r') as f:
            self.data = json.load(f)

        self.entity_mappings = self.data['entity_mappings']
        self.current_mappings = {rel: dict(mapping) for rel, mapping in self.entity_mappings.items()}

        print(f"✓ 加载了 {len(self.entity_mappings)} 个relations的mappings")

        # 统计总clusters
        total_clusters = sum(len(set(m.values())) for m in self.current_mappings.values())
        print(f"  总clusters: {total_clusters}")

        return total_clusters

    def get_clusters(self, relation: str, mappings: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        从mapping提取clusters

        Args:
            relation: relation名称
            mappings: 使用指定的mappings（默认用current_mappings）

        Returns:
            {canonical_entity: [entity1, entity2, ...]}
        """
        if mappings is None:
            mappings = self.current_mappings

        mapping = mappings[relation]
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
            max_tokens=3000,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    # ==================== 第一步: Split ====================

    def step1_split_conflicts(
        self,
        relation: str,
        backend: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> Dict[str, str]:
        """
        第一步：检查并分割语义冲突的clusters

        Args:
            relation: relation名称
            其他: LLM配置参数

        Returns:
            新的entity mapping {entity: canonical}
        """
        clusters = self.get_clusters(relation)

        system_prompt = """You are an expert in semantic analysis for knowledge graph construction.

Your task is to check if entities grouped together are semantically coherent, and split those with major conflicts.

RULES FOR SPLITTING:
1. ONLY split if there are MAJOR semantic conflicts:
   - Opposite meanings (protagonist vs antagonist)
   - Completely unrelated concepts (weapons vs beverage)
2. DO NOT split if entities are just "not identical but related"
3. Keep clusters together if they share a common theme
4. Focus on movie poster knowledge extraction context

Output ONLY the clusters that need splitting, not the ones that are fine."""

        # 构建clusters摘要
        cluster_list = []
        for canonical, entities in sorted(clusters.items()):
            if canonical.startswith('other_'):
                continue  # 跳过other
            if len(entities) == 1:
                continue  # 单个entity不需要检查
            cluster_list.append(f"  {canonical}: {', '.join(sorted(entities))}")

        if not cluster_list:
            # 没有需要检查的clusters
            return self.current_mappings[relation]

        user_prompt = f"""Relation: {relation}

Current clusters:
{chr(10).join(cluster_list)}

Please identify clusters with MAJOR semantic conflicts that should be split.

**Output Format:**
Return a JSON object:
```json
{{
  "splits": [
    {{
      "original_cluster": "detective",
      "reason": "Mixes different hero types - superhero is distinct from detective/spy",
      "new_groups": [
        {{
          "entities": ["detective", "spy", "gunslinger"],
          "suggested_name": "detective"
        }},
        {{
          "entities": ["superhero", "protagonist", "heroic_protagonist"],
          "suggested_name": "superhero"
        }}
      ]
    }}
  ]
}}
```

**Important:**
- Only include clusters that have MAJOR conflicts
- If all clusters are semantically coherent, return {{"splits": []}}
- Each entity must appear exactly once
- Provide clear reasoning for each split

Now analyze:"""

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

        # 应用splits
        new_mapping = dict(self.current_mappings[relation])

        if not result['splits']:
            return new_mapping

        for split in result['splits']:
            original = split['original_cluster']
            print(f"  Split: {original}")
            print(f"    理由: {split['reason']}")

            for group in split['new_groups']:
                group_name = group['suggested_name']
                for entity in group['entities']:
                    new_mapping[entity] = group_name
                print(f"    → {group_name}: {group['entities']}")

        return new_mapping

    # ==================== 第二步: Rename ====================

    def step2_rename_clusters(
        self,
        relation: str,
        backend: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> Dict[str, str]:
        """
        第二步：为每个cluster选择最佳的canonical name

        Args:
            relation: relation名称
            其他: LLM配置参数

        Returns:
            新的entity mapping {entity: canonical}
        """
        clusters = self.get_clusters(relation)

        system_prompt = """You are an expert in knowledge graph design.

Your task is to choose the best canonical name for each entity cluster.

RULES:
1. Prefer the most GENERIC/COMMON term (e.g., "weapon" not "darts_target")
2. Prefer shorter, simpler names
3. The name should represent the whole cluster
4. Keep "other_xxx" clusters as-is
5. Focus on semantic meaning, not alphabetical order"""

        # 构建clusters摘要
        cluster_list = []
        for canonical, entities in sorted(clusters.items()):
            if canonical.startswith('other_'):
                continue
            cluster_list.append(f"  Current name: {canonical}\n    Entities: {', '.join(sorted(entities))}")

        if not cluster_list:
            return self.current_mappings[relation]

        user_prompt = f"""Relation: {relation}

Clusters to rename:
{chr(10).join(cluster_list)}

Please choose the best canonical name for each cluster.

**Output Format:**
Return a JSON object:
```json
{{
  "renames": [
    {{
      "old_name": "darts_target",
      "entities": ["weapons", "weapon", "darts_target"],
      "new_name": "weapon",
      "reason": "weapon is more generic and represents the cluster better"
    }},
    {{
      "old_name": "detective",
      "entities": ["detective", "spy", "gunslinger"],
      "new_name": "detective",
      "reason": "already good - detective is the most representative"
    }}
  ]
}}
```

**Important:**
- Include ALL clusters (even if name doesn't change)
- Choose the most generic/representative entity as the new name
- Provide reasoning

Now choose the best names:"""

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

        # 应用renames
        new_mapping = {}

        for rename in result['renames']:
            old_name = rename['old_name']
            new_name = rename['new_name']
            entities = rename['entities']

            if old_name != new_name:
                print(f"  Rename: {old_name} → {new_name}")
                print(f"    理由: {rename['reason']}")

            for entity in entities:
                new_mapping[entity] = new_name

        # 保留other_xxx
        for entity, canonical in self.current_mappings[relation].items():
            if canonical.startswith('other_'):
                new_mapping[entity] = canonical

        return new_mapping

    # ==================== 第三步: Merge ====================

    def step3_merge_similar(
        self,
        relation: str,
        backend: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> Dict[str, str]:
        """
        第三步：根据名字合并语义相似的clusters

        Args:
            relation: relation名称
            其他: LLM配置参数

        Returns:
            新的entity mapping {entity: canonical}
        """
        clusters = self.get_clusters(relation)

        # 过滤掉other和单entity的clusters
        mergeable_clusters = {
            name: entities for name, entities in clusters.items()
            if not name.startswith('other_')
        }

        if len(mergeable_clusters) < 2:
            return self.current_mappings[relation]

        system_prompt = """You are an expert in knowledge graph design.

Your task is to identify clusters that should be merged because they represent the same or very similar concepts.

RULES:
1. ONLY merge if clusters are semantically very similar or overlapping
2. DO NOT merge just because names sound similar
3. Consider the entities in each cluster
4. Be conservative - when in doubt, don't merge
5. Keep diversity - we want 10-15 clusters per relation"""

        # 构建clusters摘要
        cluster_list = []
        for canonical, entities in sorted(mergeable_clusters.items()):
            cluster_list.append(f"  {canonical}: {', '.join(sorted(entities))}")

        user_prompt = f"""Relation: {relation}

Current clusters ({len(mergeable_clusters)} total):
{chr(10).join(cluster_list)}

Please identify clusters that should be merged.

**Output Format:**
Return a JSON object:
```json
{{
  "merges": [
    {{
      "clusters_to_merge": ["weapon", "weapons"],
      "merged_name": "weapon",
      "reason": "Both refer to the same concept - weapons/armaments"
    }},
    {{
      "clusters_to_merge": ["romantic_pair", "couple", "duo"],
      "merged_name": "couple",
      "reason": "All represent paired characters in romantic context"
    }}
  ]
}}
```

**Important:**
- Only merge semantically similar/identical clusters
- If no merges needed, return {{"merges": []}}
- Choose the best name for the merged cluster
- Provide clear reasoning

Now identify clusters to merge:"""

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

        # 应用merges
        new_mapping = dict(self.current_mappings[relation])

        if not result['merges']:
            return new_mapping

        for merge in result['merges']:
            clusters_to_merge = merge['clusters_to_merge']
            merged_name = merge['merged_name']

            print(f"  Merge: {clusters_to_merge} → {merged_name}")
            print(f"    理由: {merge['reason']}")

            # 将所有被合并的clusters指向新名字
            for entity, canonical in new_mapping.items():
                if canonical in clusters_to_merge:
                    new_mapping[entity] = merged_name

        return new_mapping

    # ==================== 主流程 ====================

    def refine_iterative(
        self,
        backend: str = 'openai',
        model_name: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        start_from_step: int = 1,
        relations_to_process: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        多轮迭代优化entity mappings

        Args:
            backend: LLM后端
            model_name: 模型名称
            api_key: API密钥
            base_url: API base URL
            temperature: 温度参数
            start_from_step: 从第几步开始（1=split, 2=rename, 3=merge）
            relations_to_process: 只处理指定的relations（None=全部）

        Returns:
            精炼后的mappings {relation: {entity: canonical}}
        """
        print(f"\n{'='*60}")
        print(f"Entity聚类迭代优化 - 三步法")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")

        relations = relations_to_process or sorted(self.current_mappings.keys())

        # 统计初始状态
        initial_counts = {
            rel: len(set(self.current_mappings[rel].values()))
            for rel in relations
        }

        # 第一步: Split
        if start_from_step <= 1:
            print(f"\n{'='*80}")
            print(f"第一步: Split - 分割语义冲突的clusters")
            print(f"{'='*80}")

            for relation in relations:
                print(f"\n[{relation}]")
                self.current_mappings[relation] = self.step1_split_conflicts(
                    relation=relation,
                    backend=backend,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature
                )

            # 保存第一步结果
            self._save_intermediate('step1_split.json', "Step 1: Split conflicts")

        # 第二步: Rename
        if start_from_step <= 2:
            print(f"\n{'='*80}")
            print(f"第二步: Rename - 选择最佳canonical names")
            print(f"{'='*80}")

            for relation in relations:
                print(f"\n[{relation}]")
                self.current_mappings[relation] = self.step2_rename_clusters(
                    relation=relation,
                    backend=backend,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature
                )

            # 保存第二步结果
            self._save_intermediate('step2_renamed.json', "Step 2: Rename clusters")

        # 第三步: Merge
        if start_from_step <= 3:
            print(f"\n{'='*80}")
            print(f"第三步: Merge - 合并相似的clusters")
            print(f"{'='*80}")

            for relation in relations:
                print(f"\n[{relation}]")
                self.current_mappings[relation] = self.step3_merge_similar(
                    relation=relation,
                    backend=backend,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature
                )

            # 保存第三步结果
            self._save_intermediate('step3_merged.json', "Step 3: Merge similar clusters")

        # 统计最终结果
        final_counts = {
            rel: len(set(self.current_mappings[rel].values()))
            for rel in relations
        }

        print(f"\n{'='*80}")
        print(f"优化完成 - 统计")
        print(f"{'='*80}")
        for rel in relations:
            before = initial_counts[rel]
            after = final_counts[rel]
            change = "→" if before != after else "="
            print(f"  {rel}: {before} {change} {after}")

        total_before = sum(initial_counts.values())
        total_after = sum(final_counts.values())
        print(f"\n  总计: {total_before} → {total_after}")

        return self.current_mappings

    def _save_intermediate(self, filename: str, step_name: str):
        """保存中间结果"""
        output_path = self.output_dir / filename

        # 统计
        total_clusters = sum(len(set(m.values())) for m in self.current_mappings.values())

        output_data = {
            'metadata': {
                **self.data.get('metadata', {}),
                'refinement_step': step_name,
                'total_clusters': total_clusters
            },
            'entity_mappings': self.current_mappings
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 中间结果已保存: {output_path}")

    def save_final_results(self, output_path: str):
        """保存最终结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 统计
        total_clusters = sum(len(set(m.values())) for m in self.current_mappings.values())

        # 统计每个relation的变化
        changes = []
        for relation in sorted(self.current_mappings.keys()):
            before = len(set(self.entity_mappings[relation].values()))
            after = len(set(self.current_mappings[relation].values()))
            if before != after:
                changes.append({
                    'relation': relation,
                    'clusters_before': before,
                    'clusters_after': after
                })

        output_data = {
            'metadata': {
                **self.data.get('metadata', {}),
                'refinement_method': 'llm_iterative_3step',
                'refinement_steps': ['split_conflicts', 'rename_clusters', 'merge_similar'],
                'llm_model': 'gpt-4o-mini',
                'total_clusters_before': sum(len(set(m.values())) for m in self.entity_mappings.values()),
                'total_clusters_after': total_clusters,
                'relations_modified': len(changes),
                'changed_relations': changes
            },
            'entity_mappings': self.current_mappings
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n{'='*80}")
        print("✅ 最终结果已保存")
        print(f"{'='*80}")
        print(f"输出文件: {output_path}")
        print(f"总clusters: {output_data['metadata']['total_clusters_before']} → {total_clusters}")
        print(f"修改的relations: {len(changes)}/{len(self.current_mappings)}")

        if changes:
            print(f"\n修改详情:")
            for item in changes:
                print(f"  {item['relation']}: {item['clusters_before']} → {item['clusters_after']}")


if __name__ == '__main__':
    print("EntityRefiner模块已加载 - 三步迭代优化")
    print("使用示例见 scripts/refine_entity_mapping.py")
