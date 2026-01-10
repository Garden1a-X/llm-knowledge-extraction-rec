"""
Entity聚类微调模块 - 使用LLM多轮优化entity聚类

三步法（正确顺序）：
1. Split - 检查并分割语义冲突的clusters
2. Merge - 基于entity本身合并相似的clusters（目标10-15）
3. Rename - 为每个cluster选择最佳的canonical name
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter


class EntityRefiner:
    """
    使用LLM多轮优化entity聚类

    工作流程（重要：顺序很关键！）：
    1. Split: 检查每个cluster，分割语义冲突的entities
    2. Merge: 基于entity本身合并语义相似的clusters（还没有独立名字的锚定）
    3. Rename: 为稳定的clusters选择最佳canonical name
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

Your task is to check if entities grouped together are semantically coherent, and split those with MAJOR conflicts.

CRITICAL: ONLY split if there are MAJOR semantic conflicts:
1. **Opposite meanings**: protagonist vs antagonist, hero vs villain
2. **Completely unrelated concepts**: weapons vs beverage, vehicles vs clothing
3. **Different semantic categories**: human vs animal, action vs object

DO NOT split if:
- Entities are similar but not identical (e.g., "dynamic_action" and "dynamic_pose" are both dynamic movements)
- Entities share a common theme or category
- They could reasonably belong to the same semantic group

Focus on movie poster knowledge extraction context."""

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

Please identify ONLY clusters with MAJOR semantic conflicts that MUST be split.

**Output Format:**
Return a JSON object:
```json
{{
  "splits": [
    {{
      "original_cluster": "antagonist",
      "reason": "Contains opposite character types - protagonist and antagonist are fundamentally opposite roles",
      "new_groups": [
        {{
          "entities": ["protagonist", "heroic_protagonist"],
          "suggested_name": "protagonist"
        }},
        {{
          "entities": ["antagonist", "villain"],
          "suggested_name": "antagonist"
        }}
      ]
    }}
  ]
}}
```

**Important:**
- ONLY include clusters with MAJOR conflicts (opposite meanings or completely unrelated)
- If all clusters are semantically coherent enough, return {{"splits": []}}
- Be very conservative - when in doubt, DON'T split
- Each entity must appear exactly once in the new groups
- Provide clear reasoning explaining why the conflict is MAJOR

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
            print(f"  无需split")
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

    # ==================== 第二步: Merge ====================

    def step2_merge_similar(
        self,
        relation: str,
        backend: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> Dict[str, str]:
        """
        第二步：基于entity本身合并语义相似的clusters

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
            print(f"  clusters < 2，无需merge")
            return self.current_mappings[relation]

        current_count = len(mergeable_clusters)

        system_prompt = """You are an expert in knowledge graph design.

Your task is to merge semantically similar or overlapping entity clusters.

IMPORTANT RULES:
1. **Target 10-15 clusters per relation** - this is critical for knowledge graph density
2. **Look at the ENTITIES themselves**, not just cluster names
3. Merge if entities represent the same or very similar semantic concepts
4. Merge entities that share common semantic themes or categories
5. Be reasonably aggressive - err on the side of merging similar concepts

Examples of what SHOULD be merged:
- "dynamic_action", "dynamic_pose", "dynamic_postures" → all dynamic movements
- "protagonist", "heroic_protagonist" → same concept
- "romantic_pair", "couple", "duo" → all paired characters

Only keep separate if entities are clearly distinct semantic categories."""

        # 构建clusters摘要（显示entities，不只是cluster名）
        cluster_list = []
        for canonical, entities in sorted(mergeable_clusters.items()):
            cluster_list.append(f"  {canonical}: [{', '.join(sorted(entities))}]")

        user_prompt = f"""Relation: {relation}

Current clusters: {current_count} (target: 10-15)
{chr(10).join(cluster_list)}

Please identify clusters that should be merged to reach ~10-15 clusters.

**Output Format:**
Return a JSON object:
```json
{{
  "merges": [
    {{
      "clusters_to_merge": ["dynamic_action", "dynamic_pose", "dynamic_postures"],
      "merged_name": "dynamic_action",
      "reason": "All represent dynamic movement/poses - same semantic category"
    }},
    {{
      "clusters_to_merge": ["protagonist", "heroic_protagonist"],
      "merged_name": "protagonist",
      "reason": "Both refer to the same concept - the main character/hero"
    }}
  ]
}}
```

**Important:**
- Focus on reaching 10-15 clusters (currently have {current_count})
- Look at the ENTITIES in brackets, not just cluster names
- Merge semantically similar/overlapping concepts
- If already in target range and well-organized, return {{"merges": []}}
- Choose the most representative entity as merged_name
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
            print(f"  无需merge")
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

    # ==================== 第三步: Rename ====================

    def step3_rename_clusters(
        self,
        relation: str,
        backend: str,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> Dict[str, str]:
        """
        第三步：为每个cluster选择最佳的canonical name

        Args:
            relation: relation名称
            其他: LLM配置参数

        Returns:
            新的entity mapping {entity: canonical}
        """
        clusters = self.get_clusters(relation)

        system_prompt = """You are an expert in knowledge graph design.

Your task is to choose the best canonical name for each finalized entity cluster.

RULES:
1. Prefer the most GENERIC/COMMON term (e.g., "weapon" not "darts_target")
2. Prefer shorter, simpler names over longer ones
3. The name should best represent ALL entities in the cluster
4. Keep "other_xxx" clusters as-is
5. Choose from the existing entities in the cluster (don't invent new names)
6. Focus on semantic representativeness, not alphabetical order"""

        # 构建clusters摘要
        cluster_list = []
        for canonical, entities in sorted(clusters.items()):
            if canonical.startswith('other_'):
                continue
            cluster_list.append(f"  Current name: {canonical}\n    Entities: {', '.join(sorted(entities))}")

        if not cluster_list:
            print(f"  无cluster需要rename")
            return self.current_mappings[relation]

        user_prompt = f"""Relation: {relation}

Finalized clusters to rename:
{chr(10).join(cluster_list)}

Please choose the best canonical name for each cluster from its entities.

**Output Format:**
Return a JSON object:
```json
{{
  "renames": [
    {{
      "old_name": "darts_target",
      "entities": ["weapons", "weapon", "darts_target"],
      "new_name": "weapon",
      "reason": "weapon is more generic and common, best represents all entities"
    }},
    {{
      "old_name": "dynamic_action",
      "entities": ["dynamic_action", "dynamic_pose", "dynamic_postures"],
      "new_name": "dynamic_action",
      "reason": "dynamic_action is most generic and covers all movement types"
    }}
  ]
}}
```

**Important:**
- Include ALL clusters (even if name stays the same)
- Choose the most generic/representative entity from the cluster
- The new_name MUST be one of the entities in the cluster
- Provide reasoning for your choice

Now choose the best canonical names:"""

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
            start_from_step: 从第几步开始（1=split, 2=merge, 3=rename）
            relations_to_process: 只处理指定的relations（None=全部）

        Returns:
            精炼后的mappings {relation: {entity: canonical}}
        """
        print(f"\n{'='*60}")
        print(f"Entity聚类迭代优化 - 三步法")
        print(f"正确顺序: Split → Merge → Rename")
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

        # 第二步: Merge
        if start_from_step <= 2:
            print(f"\n{'='*80}")
            print(f"第二步: Merge - 合并相似的clusters（基于entities本身）")
            print(f"{'='*80}")

            for relation in relations:
                print(f"\n[{relation}]")
                self.current_mappings[relation] = self.step2_merge_similar(
                    relation=relation,
                    backend=backend,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature
                )

            # 保存第二步结果
            self._save_intermediate('step2_merged.json', "Step 2: Merge similar clusters")

        # 第三步: Rename
        if start_from_step <= 3:
            print(f"\n{'='*80}")
            print(f"第三步: Rename - 选择最佳canonical names")
            print(f"{'='*80}")

            for relation in relations:
                print(f"\n[{relation}]")
                self.current_mappings[relation] = self.step3_rename_clusters(
                    relation=relation,
                    backend=backend,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature
                )

            # 保存第三步结果
            self._save_intermediate('step3_renamed.json', "Step 3: Rename clusters")

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
            in_range = "✓" if 10 <= after <= 15 else " "
            print(f"  {in_range} {rel}: {before} {change} {after}")

        total_before = sum(initial_counts.values())
        total_after = sum(final_counts.values())
        in_range_count = sum(1 for c in final_counts.values() if 10 <= c <= 15)
        print(f"\n  总计: {total_before} → {total_after}")
        print(f"  在10-15范围内: {in_range_count}/{len(relations)}")

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
                    'clusters_after': after,
                    'in_target_range': 10 <= after <= 15
                })

        output_data = {
            'metadata': {
                **self.data.get('metadata', {}),
                'refinement_method': 'llm_iterative_3step_v2',
                'refinement_steps': ['split_conflicts', 'merge_similar', 'rename_clusters'],
                'refinement_order_note': 'Split → Merge → Rename (correct order)',
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
                in_range = "✓" if item['in_target_range'] else "✗"
                print(f"  {in_range} {item['relation']}: {item['clusters_before']} → {item['clusters_after']}")


if __name__ == '__main__':
    print("EntityRefiner模块已加载 - 三步迭代优化 (Split→Merge→Rename)")
    print("使用示例见 scripts/refine_entity_mapping.py")
