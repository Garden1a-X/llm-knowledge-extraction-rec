"""
Relation聚类微调模块 - 使用LLM修正embedding聚类的语义问题
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict


class RelationRefiner:
    """
    使用LLM微调relation映射，修正embedding聚类的语义不一致问题
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
        构建LLM refinement的prompt

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
Review the current mapping and fix semantic inconsistencies. You can:
- Split overly broad standard relations into semantically coherent groups
- Merge relations if appropriate
- Create new standard relations if needed
- Move misplaced original relations to correct categories

CONSTRAINTS:
1. Focus on SEMANTIC MEANING, ignore suffixes like _type, _style, _element
2. Each standard relation should represent a single clear concept useful for movie recommendations
3. Target 12-18 standard relations (can be more or less than current {num_std_rels})
4. Preserve good mappings - don't change what's already correct""".format(
            num_std_rels=len(self.base_mapping['standard_relations'])
        )

        # 构建当前映射摘要
        mapping_summary = self._build_mapping_summary()

        # 构建问题描述
        issues_summary = self._build_issues_summary(issues) if issues else ""

        user_prompt = f"""Current relation mapping (from embedding clustering):

{mapping_summary}

{issues_summary}

Please refine this mapping to fix the semantic inconsistencies.

**CRITICAL REQUIREMENT:**
You MUST provide a COMPLETE mapping for ALL {len(self.base_mapping['relation_mapping'])} original relations.
Do not only list the changes - provide the full relation_mapping dictionary with all original relations mapped to their standard relations.

**Output Format:**
Return a JSON object with this structure:
```json
{{
  "standard_relations": [
    {{
      "name": "depicted_subject",
      "definition": "Visual entities shown in poster (characters, objects, animals, vehicles)",
      "mapped_relations": ["depicted_subject", "animal_type", "vehicle_type", ...],
      "total_instances": 260,
      "changes": "Added animal_type, vehicle_type from action_type"
    }}
  ],
  "relation_mapping": {{
    "depicted_subject": "depicted_subject",
    "animal_type": "depicted_subject",
    "vehicle_type": "depicted_subject",
    "object_type": "depicted_subject",
    "background_texture": "scene_setting",
    "background_type": "scene_setting",
    ... (INCLUDE ALL {len(self.base_mapping['relation_mapping'])} ORIGINAL RELATIONS)
  }},
  "summary": {{
    "total_changes": 15,
    "relations_before": 14,
    "relations_after": 16,
    "major_changes": [
      "Split action_type into interaction, scene_setting",
      "Separated narrative_* from genre"
    ]
  }}
}}
```

**Guidelines:**
1. Fix semantic inconsistencies - group by MEANING not word similarity
2. Each standard relation should be useful for movie recommendations
3. relation_mapping MUST contain ALL {len(self.base_mapping['relation_mapping'])} original relations
4. Provide clear reasoning for major changes

Now refine the relation mapping:"""

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
            max_tokens=4000,
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
        使用LLM微调映射

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
        print(f"使用LLM微调Relation映射")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")

        # 构建prompt
        system_prompt, user_prompt = self.build_refinement_prompt(issues)

        # 调用LLM
        print(f"\n🔄 调用LLM进行映射微调...")

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
            refined_data = json.loads(response)
            self.refined_mapping = refined_data
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始响应:\n{response}")
            raise

        # 验证结果
        self._validate_refined_mapping(refined_data)

        print(f"\n✓ LLM微调完成")
        print(f"  标准relations: {len(self.base_mapping['standard_relations'])} → "
              f"{refined_data['summary']['relations_after']}")
        print(f"  总改动数: {refined_data['summary']['total_changes']}")

        print(f"\n  主要改动:")
        for change in refined_data['summary']['major_changes']:
            print(f"    - {change}")

        return refined_data

    def _validate_refined_mapping(self, refined_data: Dict):
        """验证精炼后的映射"""
        # 检查必需字段
        required_fields = ['standard_relations', 'relation_mapping', 'summary']
        for field in required_fields:
            if field not in refined_data:
                raise ValueError(f"缺少必需字段: {field}")

        # 检查是否所有原始relations都被映射了
        original_relations = set(self.base_mapping['relation_mapping'].keys())
        mapped_relations = set(refined_data['relation_mapping'].keys())

        missing = original_relations - mapped_relations
        if missing:
            print(f"⚠️ 警告: {len(missing)}个原始relations未被映射: {missing}")

        extra = mapped_relations - original_relations
        if extra:
            print(f"⚠️ 警告: 映射中出现了额外的relations: {extra}")

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

        # 提取标准relations列表
        standard_relations = [
            item['name'] if isinstance(item, dict) else item
            for item in refined_data['standard_relations']
        ]

        # 重新计算relation_counts（基于新映射）
        new_counts = defaultdict(int)
        for orig_rel, std_rel in refined_data['relation_mapping'].items():
            count = self.base_mapping['relation_counts'].get(orig_rel, 0)
            new_counts[std_rel] += count

        # 构建输出数据
        output_data = {
            'metadata': {
                'method': 'llm_refinement_of_agglomerative',
                'base_method': self.base_mapping['metadata'].get('method', 'agglomerative'),
                'source_file': str(self.base_mapping_path),
                'total_relation_instances': self.base_mapping['metadata']['total_relation_instances'],
                'unique_relations_before': self.base_mapping['metadata']['unique_relations_before'],
                'standard_relations_after': len(standard_relations),
                'refinement_summary': refined_data.get('summary', {}),
                'embedding_model': self.base_mapping['metadata'].get('embedding_model', 'BAAI/bge-base-en-v1.5'),
                'llm_refinement_date': '2026-01-02'
            },
            'standard_relations': standard_relations,
            'relation_mapping': refined_data['relation_mapping'],
            'relation_counts': dict(self.base_mapping['relation_counts']),
            'refinement_details': refined_data.get('standard_relations', [])
        }

        # 保存
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 精炼映射已保存: {output_path}")


if __name__ == '__main__':
    print("RelationRefiner模块已加载")
    print("使用示例见 scripts/refine_relation_mapping.py")
