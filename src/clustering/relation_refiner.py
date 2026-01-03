"""
Relation聚类微调模块 - 使用LLM增量修正embedding聚类的语义问题
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict


class RelationRefiner:
    """
    使用LLM增量微调relation映射，修正embedding聚类的语义不一致问题

    核心思想：LLM只返回需要修改的部分（增量changes），不重新生成完整映射
    """

    def __init__(self, base_mapping_path: str):
        """
        初始化

        Args:
            base_mapping_path: 基础映射文件路径（embedding聚类结果）
        """
        self.base_mapping_path = Path(base_mapping_path)
        self.base_mapping = None
        self.refined_mapping = None

    def load_base_mapping(self) -> Dict:
        """
        加载基础映射（embedding聚类结果）

        Returns:
            映射数据字典
        """
        with open(self.base_mapping_path, 'r') as f:
            self.base_mapping = json.load(f)

        print(f"✓ 加载基础映射: {self.base_mapping_path}")
        print(f"  原始relations: {self.base_mapping['metadata']['unique_relations_before']}")
        print(f"  标准relations: {self.base_mapping['metadata']['standard_relations_after']}")

        return self.base_mapping

    def analyze_mapping_issues(self) -> Dict:
        """
        分析当前映射的潜在问题

        Returns:
            问题分析字典
        """
        issues = {
            'mixed_semantics': [],  # 语义混杂的标准relations
            'suffix_dominated': [],  # 被后缀主导的聚类
            'low_instances': []  # 低实例数的标准relations
        }

        # 按标准relation分组
        grouped = defaultdict(list)
        for orig_rel, std_rel in self.base_mapping['relation_mapping'].items():
            grouped[std_rel].append(orig_rel)

        # 检测问题
        for std_rel, orig_rels in grouped.items():
            # 计算总实例数
            total_instances = sum(
                self.base_mapping['relation_counts'].get(rel, 0)
                for rel in orig_rels
            )

            # 检测低实例数
            if total_instances < 20 and std_rel != 'others_relation':
                issues['low_instances'].append({
                    'standard_relation': std_rel,
                    'total_instances': total_instances,
                    'original_relations': orig_rels
                })

            # 检测后缀主导的聚类
            suffixes = [rel.split('_')[-1] for rel in orig_rels if '_' in rel]
            if len(suffixes) > 1:
                suffix_counter = Counter(suffixes)
                most_common_suffix, count = suffix_counter.most_common(1)[0]
                if count >= len(orig_rels) * 0.7:  # 70%有相同后缀
                    # 检查是否语义真的相关
                    prefixes = [rel.rsplit('_', 1)[0] for rel in orig_rels if '_' in rel]
                    unique_prefixes = set(prefixes)
                    if len(unique_prefixes) >= len(orig_rels) * 0.5:  # 50%有不同前缀
                        issues['suffix_dominated'].append({
                            'standard_relation': std_rel,
                            'common_suffix': most_common_suffix,
                            'suffix_ratio': count / len(orig_rels),
                            'original_relations': orig_rels,
                            'total_instances': total_instances
                        })

        print(f"\n✓ 映射问题分析:")
        print(f"  后缀主导的聚类: {len(issues['suffix_dominated'])}")
        print(f"  低实例数标准relations: {len(issues['low_instances'])}")

        if issues['suffix_dominated']:
            print(f"\n  后缀主导问题:")
            for issue in issues['suffix_dominated']:
                print(f"    - {issue['standard_relation']}: "
                      f"{len(issue['original_relations'])} relations, "
                      f"后缀'{issue['common_suffix']}'占{issue['suffix_ratio']*100:.0f}%")

        return issues

    def build_refinement_prompt(self, issues: Optional[Dict] = None) -> tuple[str, str]:
        """
        构建LLM增量refinement的prompt

        Args:
            issues: 分析出的问题

        Returns:
            (system_prompt, user_prompt)
        """
        system_prompt = """You are an expert in knowledge graph design for movie recommendation systems.

Your task is to refine a relation mapping that was initially created by embedding-based clustering. The clustering has semantic issues because embeddings were dominated by suffixes (_type, _style) rather than actual semantic content.

CRITICAL ISSUES TO FIX:
1. Relations with different semantic meanings were grouped together due to shared suffixes
2. Some standard relations are too broad and semantically inconsistent
3. Some relations should be split or merged based on actual semantics, not word similarity

YOUR TASK:
Review the current mapping and propose INCREMENTAL CHANGES to fix semantic inconsistencies.

IMPORTANT: You should NOT regenerate the complete mapping. Instead, specify only the changes needed using these operations:

1. **split**: Split one standard relation into multiple new ones
   - Specify which original relations go to which new standard relation

2. **merge**: Merge multiple standard relations into one
   - All original relations from sources will go to the target

3. **separate**: Move some original relations from one standard relation to another
   - Useful when a few original relations are misplaced

4. **rename**: Rename a standard relation (no redistribution of original relations)

CONSTRAINTS:
1. Focus on SEMANTIC MEANING, ignore suffixes like _type, _style, _element
2. Each standard relation should represent a single clear concept useful for movie recommendations
3. Target 12-18 standard relations (current: {num_std_rels})
4. Preserve good mappings - don't change what's already correct
5. ONLY specify changes, do NOT regenerate the entire mapping""".format(
            num_std_rels=len(self.base_mapping['standard_relations'])
        )

        # 构建当前映射摘要
        mapping_summary = self._build_mapping_summary()

        # 构建问题描述
        issues_summary = self._build_issues_summary(issues) if issues else ""

        user_prompt = f"""Current relation mapping (from embedding clustering):

{mapping_summary}

{issues_summary}

Please propose incremental changes to fix the semantic inconsistencies.

**Output Format:**
Return a JSON object with this structure:
```json
{{
  "changes": [
    {{
      "operation": "split",
      "source_standard_relation": "action_type",
      "new_standard_relations": [
        {{
          "name": "depicted_subject",
          "definition": "Visual entities shown in poster (characters, objects, animals, vehicles)",
          "receives_from_source": ["animal_type", "vehicle_type", "object_type"]
        }},
        {{
          "name": "scene_setting",
          "definition": "Environment and location depicted",
          "receives_from_source": ["setting_type", "landscape_type"]
        }}
      ],
      "reason": "action_type mixed unrelated concepts - objects/animals are not actions"
    }},
    {{
      "operation": "merge",
      "source_standard_relations": ["character_type", "person_depicted"],
      "target_standard_relation": {{
        "name": "character_profile",
        "definition": "Characters and people shown in the poster"
      }},
      "reason": "Both describe the same concept - people/characters in the poster"
    }},
    {{
      "operation": "separate",
      "source_standard_relation": "genre",
      "original_relations_to_move": ["narrative_structure", "narrative_tone"],
      "target_standard_relation": {{
        "name": "narrative_style",
        "definition": "How the story is told",
        "is_new": true
      }},
      "reason": "Narrative structure/tone are different from genre classification"
    }},
    {{
      "operation": "rename",
      "old_name": "text_type",
      "new_name": "text_content",
      "reason": "More descriptive name"
    }}
  ],
  "summary": {{
    "total_changes": 4,
    "relations_before": {len(self.base_mapping['standard_relations'])},
    "relations_after": 16,
    "major_improvements": [
      "Separated visual subjects from actions",
      "Unified character-related relations"
    ]
  }}
}}
```

**Guidelines:**
1. Only specify changes needed, not the complete mapping
2. For "split": list which original relations go to which new standard relation
3. For "merge": specify source relations and target relation name
4. For "separate": specify which original relations to move and where
5. For "rename": just provide old and new names
6. Ensure every original relation remains mapped after changes
7. Provide clear reasoning for each change

Now propose the incremental changes:"""

        return system_prompt, user_prompt

    def _build_mapping_summary(self) -> str:
        """构建当前映射的摘要"""
        lines = []

        # 按标准relation分组
        grouped = defaultdict(list)
        for orig_rel, std_rel in self.base_mapping['relation_mapping'].items():
            grouped[std_rel].append(orig_rel)

        # 为每个标准relation生成摘要
        for std_rel in sorted(grouped.keys()):
            orig_rels = grouped[std_rel]
            total_instances = sum(
                self.base_mapping['relation_counts'].get(rel, 0)
                for rel in orig_rels
            )

            # 限制显示的原始relations数量
            if len(orig_rels) > 8:
                shown = orig_rels[:5] + ['...'] + orig_rels[-3:]
            else:
                shown = orig_rels

            lines.append(f"{std_rel} ({total_instances} instances):")
            lines.append(f"  {', '.join(shown)}")
            lines.append("")

        return '\n'.join(lines)

    def _build_issues_summary(self, issues: Dict) -> str:
        """构建问题描述"""
        if not issues:
            return ""

        lines = ["IDENTIFIED ISSUES:\n"]

        # 后缀主导问题
        if issues['suffix_dominated']:
            for idx, issue in enumerate(issues['suffix_dominated'], 1):
                std_rel = issue['standard_relation']
                orig_rels = issue['original_relations']
                suffix = issue['common_suffix']

                lines.append(f"{idx}. {std_rel} is suffix-dominated ('{suffix}'):")
                lines.append(f"   Contains: {', '.join(orig_rels[:8])}")
                lines.append(f"   Problem: Relations grouped by '{suffix}' suffix, not semantic meaning")
                lines.append("")

        # 低实例数问题
        if issues['low_instances']:
            lines.append(f"\nLow instance standard relations (< 20 instances):")
            for issue in issues['low_instances']:
                lines.append(f"  - {issue['standard_relation']}: "
                           f"{issue['total_instances']} instances")

        return '\n'.join(lines)

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
            max_tokens=4000,  # 增量changes不需要太多tokens
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    def refine_with_llm(
        self,
        backend: str = 'openai',
        model_name: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        issues: Optional[Dict] = None
    ) -> Dict:
        """
        使用LLM增量微调映射

        Args:
            backend: LLM后端
            model_name: 模型名称
            api_key: API密钥
            base_url: API base URL
            temperature: 温度参数
            issues: 分析出的问题

        Returns:
            精炼后的映射
        """
        print(f"\n{'='*60}")
        print(f"使用LLM增量微调Relation映射")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")

        # 构建prompt
        system_prompt, user_prompt = self.build_refinement_prompt(issues)

        # 调用LLM
        print(f"\n🔄 调用LLM获取增量修改建议...")

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

        # 解析JSON响应
        try:
            changes_data = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始响应:\n{response}")
            raise

        # 应用增量修改
        print(f"\n🔄 应用增量修改...")
        refined_mapping = self._apply_incremental_changes(changes_data)

        self.refined_mapping = refined_mapping

        print(f"\n✓ LLM微调完成")
        print(f"  标准relations: {len(self.base_mapping['standard_relations'])} → "
              f"{refined_mapping['summary']['relations_after']}")
        print(f"  总改动数: {refined_mapping['summary']['total_changes']}")

        if 'major_improvements' in refined_mapping['summary']:
            print(f"\n  主要改进:")
            for improvement in refined_mapping['summary']['major_improvements']:
                print(f"    - {improvement}")

        return refined_mapping

    def _apply_incremental_changes(self, changes_data: Dict) -> Dict:
        """
        应用增量修改到base_mapping

        Args:
            changes_data: LLM返回的changes数据

        Returns:
            精炼后的映射数据
        """
        # 复制base mapping作为工作副本
        new_relation_mapping = dict(self.base_mapping['relation_mapping'])
        new_standard_relations = list(self.base_mapping['standard_relations'])

        change_log = []

        for change in changes_data['changes']:
            operation = change['operation']

            if operation == 'split':
                # Split操作: 将一个standard relation分成多个
                source = change['source_standard_relation']

                # 移除旧的standard relation
                if source in new_standard_relations:
                    new_standard_relations.remove(source)
                    change_log.append(f"Removed standard relation: {source}")

                # 添加新的standard relations
                for new_std in change['new_standard_relations']:
                    new_name = new_std['name']
                    new_standard_relations.append(new_name)
                    change_log.append(f"Added standard relation: {new_name} - {new_std['definition']}")

                    # 重新映射指定的original relations
                    for orig_rel in new_std['receives_from_source']:
                        if orig_rel in new_relation_mapping and new_relation_mapping[orig_rel] == source:
                            new_relation_mapping[orig_rel] = new_name
                            change_log.append(f"  Moved {orig_rel}: {source} → {new_name}")

            elif operation == 'merge':
                # Merge操作: 合并多个standard relations
                sources = change['source_standard_relations']
                target = change['target_standard_relation']
                target_name = target['name'] if isinstance(target, dict) else target

                # 移除旧的standard relations
                for source in sources:
                    if source in new_standard_relations:
                        new_standard_relations.remove(source)
                        change_log.append(f"Removed standard relation: {source}")

                # 添加新的target relation（如果不存在）
                if target_name not in new_standard_relations:
                    new_standard_relations.append(target_name)
                    if isinstance(target, dict):
                        change_log.append(f"Added standard relation: {target_name} - {target['definition']}")
                    else:
                        change_log.append(f"Added standard relation: {target_name}")

                # 重新映射所有从sources来的original relations
                for orig_rel, std_rel in new_relation_mapping.items():
                    if std_rel in sources:
                        new_relation_mapping[orig_rel] = target_name
                        change_log.append(f"  Moved {orig_rel}: {std_rel} → {target_name}")

            elif operation == 'separate':
                # Separate操作: 从一个standard relation中移出部分original relations
                source = change['source_standard_relation']
                to_move = change['original_relations_to_move']
                target = change['target_standard_relation']
                target_name = target['name'] if isinstance(target, dict) else target

                # 添加target relation（如果是新的）
                if target.get('is_new', False) and target_name not in new_standard_relations:
                    new_standard_relations.append(target_name)
                    change_log.append(f"Added standard relation: {target_name} - {target['definition']}")

                # 移动指定的original relations
                for orig_rel in to_move:
                    if orig_rel in new_relation_mapping and new_relation_mapping[orig_rel] == source:
                        new_relation_mapping[orig_rel] = target_name
                        change_log.append(f"  Moved {orig_rel}: {source} → {target_name}")

            elif operation == 'rename':
                # Rename操作: 重命名standard relation
                old_name = change['old_name']
                new_name = change['new_name']

                # 更新standard relations列表
                if old_name in new_standard_relations:
                    idx = new_standard_relations.index(old_name)
                    new_standard_relations[idx] = new_name
                    change_log.append(f"Renamed standard relation: {old_name} → {new_name}")

                # 更新所有映射
                for orig_rel, std_rel in new_relation_mapping.items():
                    if std_rel == old_name:
                        new_relation_mapping[orig_rel] = new_name

        # 验证结果
        self._validate_incremental_result(new_relation_mapping, new_standard_relations, change_log)

        # 构建返回数据
        refined_data = {
            'standard_relations': new_standard_relations,
            'relation_mapping': new_relation_mapping,
            'summary': changes_data.get('summary', {
                'total_changes': len(changes_data['changes']),
                'relations_before': len(self.base_mapping['standard_relations']),
                'relations_after': len(new_standard_relations)
            }),
            'change_log': change_log,
            'original_changes': changes_data['changes']
        }

        return refined_data

    def _validate_incremental_result(
        self,
        new_relation_mapping: Dict[str, str],
        new_standard_relations: List[str],
        change_log: List[str]
    ):
        """
        验证增量修改的结果

        Args:
            new_relation_mapping: 新的relation映射
            new_standard_relations: 新的标准relations列表
            change_log: 修改日志
        """
        # 检查1: 所有原始relations都还在
        original_relations = set(self.base_mapping['relation_mapping'].keys())
        mapped_relations = set(new_relation_mapping.keys())

        if original_relations != mapped_relations:
            missing = original_relations - mapped_relations
            extra = mapped_relations - original_relations
            if missing:
                raise ValueError(f"错误: {len(missing)}个原始relations丢失: {list(missing)[:5]}")
            if extra:
                raise ValueError(f"错误: {len(extra)}个额外relations出现: {list(extra)[:5]}")

        # 检查2: 没有orphan mappings（所有映射的值都在standard_relations中）
        std_rel_set = set(new_standard_relations)
        orphans = []
        for orig_rel, std_rel in new_relation_mapping.items():
            if std_rel not in std_rel_set:
                orphans.append((orig_rel, std_rel))

        if orphans:
            print(f"\n❌ 发现 {len(orphans)} 个orphan映射:")
            for orig_rel, std_rel in orphans[:10]:
                print(f"  {orig_rel} → {std_rel} (不在standard_relations中)")
            raise ValueError(f"验证失败: 存在{len(orphans)}个orphan映射")

        # 检查3: 所有standard relations都被使用
        used_std_rels = set(new_relation_mapping.values())
        unused = std_rel_set - used_std_rels
        if unused:
            print(f"\n⚠️  警告: {len(unused)}个标准relations未被使用: {list(unused)}")

        print(f"\n✓ 验证通过:")
        print(f"  原始relations: {len(original_relations)}个全部保留")
        print(f"  标准relations: {len(new_standard_relations)}个，全部有效")
        print(f"  无orphan映射")

    def save_refined_mapping(
        self,
        output_path: str,
        refined_data: Optional[Dict] = None
    ):
        """
        保存精炼后的映射

        Args:
            output_path: 输出路径
            refined_data: 精炼后的数据（如果为None，使用self.refined_mapping）
        """
        if refined_data is None:
            refined_data = self.refined_mapping

        if refined_data is None:
            raise ValueError("没有可保存的精炼映射，请先运行 refine_with_llm()")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 重新计算relation_counts（基于新映射）
        new_counts = defaultdict(int)
        for orig_rel, std_rel in refined_data['relation_mapping'].items():
            count = self.base_mapping['relation_counts'].get(orig_rel, 0)
            new_counts[std_rel] += count

        # 构建输出数据
        output_data = {
            'metadata': {
                'method': 'llm_incremental_refinement',
                'base_method': self.base_mapping['metadata'].get('method', 'agglomerative'),
                'source_file': str(self.base_mapping_path),
                'total_relation_instances': self.base_mapping['metadata']['total_relation_instances'],
                'unique_relations_before': self.base_mapping['metadata']['unique_relations_before'],
                'standard_relations_after': len(refined_data['standard_relations']),
                'refinement_summary': refined_data.get('summary', {}),
                'embedding_model': self.base_mapping['metadata'].get('embedding_model', 'BAAI/bge-base-en-v1.5'),
                'llm_refinement_date': '2026-01-02'
            },
            'standard_relations': refined_data['standard_relations'],
            'relation_mapping': refined_data['relation_mapping'],
            'relation_counts': dict(self.base_mapping['relation_counts']),
            'standard_relation_counts': dict(new_counts),
            'refinement_details': {
                'changes': refined_data.get('original_changes', []),
                'change_log': refined_data.get('change_log', [])
            }
        }

        # 保存
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 精炼映射已保存: {output_path}")


if __name__ == '__main__':
    print("RelationRefiner模块已加载（增量微调版本）")
    print("使用示例见 scripts/refine_relation_mapping.py")
