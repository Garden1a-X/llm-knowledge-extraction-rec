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
            max_tokens=12000,  # 增加到12000以支持超大relation的CoT分析（如text_style有20+entities/cluster）
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

        system_prompt = """You are an expert in knowledge graph design for MOVIE RECOMMENDATION SYSTEMS.

**BUSINESS CONTEXT:**
We are building a movie poster knowledge graph for recommendation. Each entity represents a visual or thematic attribute that:
1. Users may have preferences about (e.g., "I like movies with protagonists" vs "I like movies with villains")
2. Influences whether users choose to watch a movie
3. Helps match users with movies based on their interests

**YOUR TASK:**
Check if entities grouped together are semantically coherent for recommendation purposes, and split those with MAJOR conflicts.

**CLUSTERING PRINCIPLES FOR RECOMMENDATION:**
1. **Same cluster = Similar user preference**: Entities in the same cluster should appeal to similar user interests
   - Example: "protagonist" and "heroic_protagonist" appeal to the same preference
2. **Different clusters = Different user preferences**: Clusters should represent DISTINCT preferences
   - Example: "protagonist" (hero preference) vs "antagonist" (villain preference) are OPPOSITE
3. **Discriminative power**: Clusters must help distinguish different types of movies
   - Example: In character_type, we need to tell hero movies from villain movies

**CRITICAL: ONLY split if there are MAJOR conflicts:**
1. **Opposite user preferences**: protagonist vs antagonist (people who like heroes ≠ people who like villains)
2. **Completely unrelated interests**: weapons vs beverage (different aspects of interest)
3. **Different semantic categories**: human vs animal, action vs object

**DO NOT split if:**
- Entities appeal to similar user interests (e.g., "dynamic_action" and "dynamic_pose" both appeal to "dynamic movement")
- Entities share a common theme or category
- They could reasonably belong to the same user preference

**CRITICAL: Consider the RELATION context**
- You are analyzing entities within a specific RELATION (e.g., character_type, mood, color_palette)
- The relation defines the semantic framework - all entities should make sense within this relation
- Consider: Would users who like one entity in the cluster also like the others?"""

        # 构建全局上下文：显示所有clusters
        all_clusters_list = []
        for canonical, entities in sorted(clusters.items()):
            if canonical.startswith('other_'):
                continue
            all_clusters_list.append(f"  {canonical}: [{', '.join(sorted(entities))}]")

        # 构建需要检查的clusters列表
        check_list = []
        for canonical, entities in sorted(clusters.items()):
            if canonical.startswith('other_'):
                continue
            if len(entities) == 1:
                continue
            check_list.append(f"  {canonical}: {', '.join(sorted(entities))}")

        if not check_list:
            # 没有需要检查的clusters
            return self.current_mappings[relation]

        user_prompt = f"""Relation: **{relation}** (defines the semantic framework for recommendation)

**RECOMMENDATION CONTEXT:**
This relation helps users find movies based on their preferences for "{relation}".
- Users may prefer certain types within this relation (e.g., in character_type: some like hero movies, others prefer complex villain movies)
- Clusters must represent DISTINCT user preferences to be useful for recommendation

**GLOBAL CONTEXT - All current clusters in this relation:**
{chr(10).join(all_clusters_list)}

**Clusters to check for conflicts:**
{chr(10).join(check_list)}

Please identify ONLY clusters with MAJOR semantic conflicts that MUST be split for recommendation purposes.

**ANALYSIS PROCESS (complete this BEFORE making split decisions):**
For each cluster with 2+ entities, analyze:
1. Map each entity to its user preference
2. Check if preferences conflict (opposite or completely different)
3. Assess recommendation impact (does mixing lose discriminative power?)
4. Make decision: SPLIT or KEEP

**Output Format (先完成analysis，再给出splits):**
Return a JSON object:
```json
{{
  "cluster_analysis": {{
    "villain": {{
      "entities": ["antagonist", "detective", "spy", "superhero", "villain"],
      "user_preference_mapping": {{
        "superhero": "hero/protagonist preference - users who like heroic characters saving the day",
        "detective": "investigation preference - users who like detective/mystery themes",
        "spy": "espionage preference - users who like spy/action themes",
        "antagonist": "villain preference - users who like antagonist/villain characters",
        "villain": "villain preference - users who like villain characters"
      }},
      "conflict_check": "superhero (hero preference) vs villain/antagonist (villain preference) are OPPOSITE. detective/spy are different from both.",
      "recommendation_impact": "Mixing heroes and villains completely loses discriminative power - cannot distinguish hero movies from villain movies",
      "decision": "MUST SPLIT into 3 groups"
    }},
    "dynamic_action": {{
      "entities": ["dynamic_action", "dynamic_pose"],
      "user_preference_mapping": {{
        "dynamic_action": "users who like dynamic/action movement",
        "dynamic_pose": "users who like dynamic poses"
      }},
      "conflict_check": "Both appeal to same 'dynamic movement' preference",
      "recommendation_impact": "No conflict - both represent similar user interest",
      "decision": "KEEP TOGETHER"
    }}
  }},
  "splits": [
    {{
      "original_cluster": "villain",
      "reason": "Based on analysis: Contains 3 DIFFERENT user preferences (hero vs villain vs investigation). Opposite preferences (hero/villain) MUST be separated for recommendation.",
      "new_groups": [
        {{
          "entities": ["superhero"],
          "suggested_name": "superhero"
        }},
        {{
          "entities": ["detective", "spy"],
          "suggested_name": "detective"
        }},
        {{
          "entities": ["antagonist", "villain"],
          "suggested_name": "villain"
        }}
      ]
    }}
  ]
}}
```

**Important:**
- FIRST complete cluster_analysis for ALL clusters with 2+ entities
- THEN output splits array based on analysis
- ONLY split if analysis shows MAJOR conflicts (opposite preferences, unrelated concepts)
- If all clusters are coherent, return {{"cluster_analysis": {{}}, "splits": []}}
- Be very conservative - when in doubt, decision should be "KEEP TOGETHER"

Now analyze step by step:"""

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
            print(f"原始响应长度: {len(response)} 字符")

            # 尝试简单的JSON修复：如果是截断问题，尝试补全
            if "Expecting property name" in str(e) or "Unterminated string" in str(e):
                print(f"⚠️  疑似JSON被截断（可能超出max_tokens限制）")
                print(f"  尝试优雅降级：保持{relation}不变")
                # 返回原mapping，不做修改
                return self.current_mappings[relation]

            print(f"原始响应:\n{response}")
            raise

        # 验证response结构
        if 'splits' not in result:
            print(f"⚠️  警告: LLM返回的JSON缺少'splits'字段")
            print(f"原始响应:\n{response}")
            # 尝试从cluster_analysis推断是否需要split
            if 'cluster_analysis' in result:
                print(f"  包含cluster_analysis，但没有splits数组，视为无需split")
            result['splits'] = []

        # 应用splits
        new_mapping = dict(self.current_mappings[relation])

        if not result['splits']:
            print(f"  无需split")
            return new_mapping

        for split in result['splits']:
            # 验证split对象结构
            if 'original_cluster' not in split or 'new_groups' not in split:
                print(f"⚠️  警告: split对象缺少必需字段，跳过")
                print(f"  split对象: {split}")
                continue

            original = split['original_cluster']
            print(f"  Split: {original}")
            if 'reason' in split:
                print(f"    理由: {split['reason']}")

            for group in split['new_groups']:
                # 验证group对象结构
                if 'entities' not in group or 'suggested_name' not in group:
                    print(f"⚠️  警告: new_group缺少必需字段，跳过")
                    print(f"  group对象: {group}")
                    continue

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

        system_prompt = """You are an expert in knowledge graph design for MOVIE RECOMMENDATION SYSTEMS.

**BUSINESS CONTEXT:**
We are building a movie poster knowledge graph for recommendation. The goal is to match users with movies based on their preferences.

**YOUR TASK:**
Merge semantically similar clusters to create 10-15 meaningful categories per relation that users can have preferences about.

**CLUSTERING PRINCIPLES FOR RECOMMENDATION:**
1. **Same cluster = Same user preference**: Only merge if entities appeal to the SAME user interests
   - Example: "protagonist" + "heroic_protagonist" → both appeal to "hero movie fans"
   - Example: "dynamic_action" + "dynamic_pose" → both appeal to "dynamic movement fans"

2. **Different clusters = Different user preferences**: Keep DISTINCT if they represent different preferences
   - Example: "protagonist" vs "antagonist" → hero fans ≠ villain fans (NEVER merge!)
   - Example: "joyful" vs "ominous" → different mood preferences (NEVER merge!)

3. **Discriminative power**: Merged clusters must still help distinguish movie types
   - If merging loses important distinctions for recommendation, DON'T merge
   - Example: Merging all character types into one cluster loses all character-based recommendation power

4. **Target 10-15 clusters per relation** - optimal for knowledge graph density without losing discriminative power

**IMPORTANT RULES:**
1. **Look at ENTITIES themselves**, not just cluster names
2. Merge if entities appeal to similar user interests
3. Be reasonably aggressive - err on the side of merging similar concepts
4. But NEVER merge opposite or contradictory user preferences

Examples of what SHOULD be merged:
- "romantic_pair", "couple", "duo" → all about paired characters (same preference)
- "vibrant_colors", "bright_colors" → all about vivid color preference

Examples of what should NEVER be merged:
- "protagonist" + "antagonist" → opposite character preferences
- "horror" + "comedy" → opposite genre preferences
- "joyful" + "somber" → opposite mood preferences

**CRITICAL: Consider GLOBAL CONTEXT**
- You are merging clusters within a specific RELATION (e.g., character_type, mood)
- The relation defines the semantic framework - all clusters should fit coherently within it
- Consider the OVERALL structure - does the final set represent distinct user preferences?
- Ask: Would users who prefer cluster A have DIFFERENT preferences from users who prefer cluster B?"""

        # 构建clusters摘要（显示entities，不只是cluster名）
        cluster_list = []
        for canonical, entities in sorted(mergeable_clusters.items()):
            cluster_list.append(f"  {canonical}: [{', '.join(sorted(entities))}]")

        user_prompt = f"""Relation: **{relation}** (defines how users find movies by {relation} preferences)

**RECOMMENDATION CONTEXT:**
Users have preferences for "{relation}" attributes. We need 10-15 distinct clusters representing different user preferences.
- Current count: {current_count} clusters (target: 10-15)

**GLOBAL CONTEXT - All current clusters:**
{chr(10).join(cluster_list)}

Please identify clusters that should be merged to create meaningful, distinct preference categories.

**ANALYSIS PROCESS (complete this BEFORE making merge decisions):**
For each potential merge pair, analyze:
1. Check entities in both clusters - what user preferences do they represent?
2. Same preference check: Would users who like A also like B?
3. Opposite check: Are they opposite/contradictory preferences?
4. Discriminative power check: Does merging lose important distinctions?
5. Make decision: MERGE or KEEP SEPARATE

**Output Format (先完成analysis，再给出merges):**
Return a JSON object:
```json
{{
  "merge_analysis": {{
    "potential_merge_1": {{
      "clusters": ["dynamic_action", "dynamic_pose"],
      "entities_in_clusters": {{
        "dynamic_action": ["dynamic_action", "dynamic_stance"],
        "dynamic_pose": ["dynamic_pose", "dynamic_postures"]
      }},
      "user_preference_check": "dynamic_action appeals to 'action movement fans'; dynamic_pose appeals to 'dynamic pose fans' → SAME underlying preference for dynamic/active scenes",
      "opposite_check": "NOT opposite - both represent active/dynamic preference (not static vs dynamic)",
      "discriminative_check": "Merging maintains discriminative power - still distinct from static/calm scenes",
      "decision": "SHOULD MERGE",
      "merged_name": "dynamic_action",
      "reason": "Both serve same user preference for dynamic movement"
    }},
    "potential_merge_2": {{
      "clusters": ["protagonist", "antagonist"],
      "entities_in_clusters": {{
        "protagonist": ["protagonist", "heroic_protagonist"],
        "antagonist": ["antagonist", "villain"]
      }},
      "user_preference_check": "protagonist appeals to 'hero movie fans'; antagonist appeals to 'villain movie fans' → DIFFERENT and OPPOSITE preferences",
      "opposite_check": "YES - hero preference vs villain preference are OPPOSITE",
      "discriminative_check": "These are KEY distinctions for recommendation - MUST keep separate",
      "decision": "MUST NOT MERGE",
      "merged_name": null,
      "reason": "Opposite user preferences - hero fans ≠ villain fans"
    }}
  }},
  "merges": [
    {{
      "clusters_to_merge": ["dynamic_action", "dynamic_pose"],
      "merged_name": "dynamic_action",
      "reason": "Based on analysis: Same user preference for dynamic movement. Merging maintains discriminative power."
    }}
  ]
}}
```

**Important:**
- FIRST complete merge_analysis for ALL potential merge pairs you consider
- Include both SHOULD MERGE and MUST NOT MERGE cases in analysis (shows reasoning)
- THEN output merges array containing only the ones that SHOULD MERGE
- Focus on reaching 10-15 clusters (currently have {current_count})
- NEVER merge opposite preferences (hero+villain, horror+comedy, joyful+somber, protagonist+antagonist)
- If already in target range and well-organized, return {{"merge_analysis": {{}}, "merges": []}}

Now analyze step by step:"""

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

        # 验证response结构
        if 'merges' not in result:
            print(f"⚠️  警告: LLM返回的JSON缺少'merges'字段")
            print(f"原始响应:\n{response}")
            if 'merge_analysis' in result:
                print(f"  包含merge_analysis，但没有merges数组，视为无需merge")
            result['merges'] = []

        # 应用merges
        new_mapping = dict(self.current_mappings[relation])

        if not result['merges']:
            print(f"  无需merge")
            return new_mapping

        for merge in result['merges']:
            # 验证merge对象结构
            if 'clusters_to_merge' not in merge or 'merged_name' not in merge:
                print(f"⚠️  警告: merge对象缺少必需字段，跳过")
                print(f"  merge对象: {merge}")
                continue

            clusters_to_merge = merge['clusters_to_merge']
            merged_name = merge['merged_name']

            print(f"  Merge: {clusters_to_merge} → {merged_name}")
            if 'reason' in merge:
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

        system_prompt = """You are an expert in knowledge graph design for MOVIE RECOMMENDATION SYSTEMS.

**BUSINESS CONTEXT:**
We are building a movie poster knowledge graph for recommendation. Cluster names will be used to:
1. Help users express preferences (e.g., "I like movies with protagonists")
2. Match users with movies based on these attributes
3. Describe movie characteristics in the recommendation interface

**YOUR TASK:**
Choose the best canonical name for each cluster that users will understand and use for preferences.

**NAMING PRINCIPLES FOR RECOMMENDATION:**
1. **User-friendly**: Name should be clear and understandable to users
2. **Generic/Common**: Prefer the most GENERIC/COMMON term (e.g., "weapon" not "darts_target")
3. **Representative**: Name should represent ALL entities in the cluster
4. **Distinct**: Names must be DIFFERENT across clusters to represent different preferences
5. **Simple**: Prefer shorter, simpler names over longer ones
6. **Semantic**: Choose from existing entities in the cluster (don't invent new names)

**CRITICAL: Consider GLOBAL CONTEXT**
- You are naming clusters within a specific RELATION (e.g., character_type, mood)
- Names must be DISTINCT - they represent DIFFERENT user preferences
- Names should form a coherent naming system for the relation
- Avoid duplicate or confusingly similar names
- Example: In "character_type", if you have "protagonist" for heroes, DON'T name the villain cluster as "protagonist"
- Example: In "genre", "horror" and "comedy" must be clearly distinct names"""

        # 构建全局上下文：显示所有cluster名字
        all_cluster_names = [name for name in sorted(clusters.keys()) if not name.startswith('other_')]

        # 构建clusters摘要
        cluster_list = []
        for canonical, entities in sorted(clusters.items()):
            if canonical.startswith('other_'):
                continue
            cluster_list.append(f"  Current name: {canonical}\n    Entities: {', '.join(sorted(entities))}")

        if not cluster_list:
            print(f"  无cluster需要rename")
            return self.current_mappings[relation]

        user_prompt = f"""Relation: **{relation}** (defines how users express preferences for {relation})

**RECOMMENDATION CONTEXT:**
Users will use these cluster names to:
- Express preferences: "I like movies with [cluster_name]"
- Search and filter movies: "Show me movies with [cluster_name]"
- Understand movie characteristics in recommendations

**GLOBAL CONTEXT - All cluster names in this relation:**
{', '.join(all_cluster_names)}
(Names must be DISTINCT and represent DIFFERENT user preferences)

**Finalized clusters to rename:**
{chr(10).join(cluster_list)}

Please choose the best canonical name for each cluster that clearly represents its USER PREFERENCE role.

**ANALYSIS PROCESS (complete this BEFORE choosing names):**
For each cluster, analyze:
1. What user preference does this cluster represent?
2. Which entity is most user-friendly and generic?
3. Is this name distinct from other cluster names?
4. Does it fit the relation's semantic framework?
5. Choose the best name

**Output Format (先完成analysis，再给出renames):**
Return a JSON object:
```json
{{
  "naming_analysis": {{
    "darts_target": {{
      "current_name": "darts_target",
      "entities": ["weapons", "weapon", "darts_target"],
      "user_preference": "Users who prefer movies with weapons/combat elements",
      "user_friendly_check": "weapon (✓ common) vs darts_target (✗ too specific) vs weapons (✓ generic plural)",
      "generic_check": "weapon is most generic singular form",
      "distinctness_check": "weapon is distinct from other clusters: beverage, background, character_interaction, etc.",
      "relation_fit": "Fits additional_elements as a visual object category",
      "recommended_name": "weapon",
      "reasoning": "weapon is most generic, user-friendly, and distinct"
    }},
    "villain": {{
      "current_name": "villain",
      "entities": ["antagonist", "villain"],
      "user_preference": "Users who prefer movies with villain/antagonist characters",
      "user_friendly_check": "Both villain and antagonist are user-friendly",
      "generic_check": "villain is more common in everyday language",
      "distinctness_check": "Must be distinct from 'protagonist' or 'hero' clusters (opposite preferences)",
      "relation_fit": "Fits character_type as character role category",
      "recommended_name": "villain",
      "reasoning": "villain is more generic and clearly opposite to hero/protagonist"
    }}
  }},
  "renames": [
    {{
      "old_name": "darts_target",
      "entities": ["weapons", "weapon", "darts_target"],
      "new_name": "weapon",
      "reason": "Based on analysis: weapon is most generic and user-friendly, distinct from other clusters"
    }},
    {{
      "old_name": "villain",
      "entities": ["antagonist", "villain"],
      "new_name": "villain",
      "reason": "Based on analysis: villain is more common, clearly distinct from hero/protagonist"
    }}
  ]
}}
```

**Important:**
- FIRST complete naming_analysis for ALL clusters
- Show reasoning for each naming choice in analysis
- THEN output renames array with final decisions
- Include ALL clusters (even if name stays the same)
- The new_name MUST be one of the entities in the cluster
- Names must be DISTINCT - never conflate opposite preferences (hero≠villain, horror≠comedy, joyful≠somber)
- Choose most generic/user-friendly option

Now analyze and choose names step by step:"""

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

        # 验证response结构
        if 'renames' not in result:
            print(f"⚠️  警告: LLM返回的JSON缺少'renames'字段")
            print(f"原始响应:\n{response}")
            if 'naming_analysis' in result:
                print(f"  包含naming_analysis，但没有renames数组，保持原名")
            # 保持原名不变
            return self.current_mappings[relation]

        # 应用renames
        new_mapping = {}

        # 获取当前clusters用于推断缺失的entities
        current_clusters = self.get_clusters(relation)

        for rename in result['renames']:
            # 验证rename对象基本结构
            if 'old_name' not in rename or 'new_name' not in rename:
                print(f"⚠️  警告: rename对象缺少old_name或new_name，跳过")
                print(f"  rename对象: {rename}")
                continue

            old_name = rename['old_name']
            new_name = rename['new_name']

            # 如果entities缺失，从当前clusters推断
            if 'entities' not in rename:
                if old_name in current_clusters:
                    entities = current_clusters[old_name]
                    print(f"  ℹ️  从当前clusters推断entities: {old_name} → {entities}")
                else:
                    print(f"⚠️  警告: 无法推断entities，old_name '{old_name}' 不在当前clusters中，跳过")
                    continue
            else:
                entities = rename['entities']

            if old_name != new_name:
                print(f"  Rename: {old_name} → {new_name}")
                if 'reason' in rename:
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
