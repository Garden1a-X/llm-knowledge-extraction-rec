#!/usr/bin/env python3
"""
Phase 2b-0: Entity重分配（两阶段LLM处理）

目标：
1. 第一轮：按relation清理entities（剔除不属于的 + 合并同义词）
2. 第二轮：重分配孤儿entities到正确的relation

使用：
  python scripts/phase2b_entity_redistribution.py --relation visual_theme --dry-run
  python scripts/phase2b_entity_redistribution.py --all
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import os
import sys
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.mllm_interface import create_mllm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
DOCS_DIR = PROJECT_ROOT / 'docs'

# Relation定义（从RELATION_DEFINITIONS.md提取）
RELATION_DEFINITIONS = {
    "depicted_subject": {
        "description": "画面的主要视觉主体（what is shown）",
        "includes": [
            "人物基础类型：man, woman, child, face, human_figure",
            "非人物对象：building, car, weapon, statue, animal, dog",
            "场景：landscape, cityscape, ocean, sky"
        ],
        "excludes": [
            "人物的角色/职业/关系 → character_type"
        ],
        "boundary_rule": "如果是人物+有角色属性→character_type；如果是人物+仅基础描述→depicted_subject；所有非人物→depicted_subject"
    },

    "character_type": {
        "description": "人物的角色类型、职业、身份、关系（仅限人物相关）",
        "includes": [
            "职业/角色：warrior, detective, hero, villain, cowboy",
            "关系：couple, romantic_couple, family, group",
            "社会身份：outlaw, royalty, civilian"
        ],
        "excludes": [
            "基础人物类型（man, woman, child） → depicted_subject",
            "非人物 → depicted_subject"
        ],
        "boundary_rule": "如果entity明确表达人物的角色/职业/关系→character_type；如果仅描述人物基础类型/数量/性别→depicted_subject"
    },

    "visual_theme": {
        "description": "整体视觉/叙事主题、概念性主题",
        "includes": [
            "叙事主题：war, romance, adventure, survival, coming_of_age",
            "概念主题：urban_life, nature, technology, isolation, freedom",
            "视觉概念：minimalism, surrealism, realism, abstraction"
        ],
        "excludes": [
            "单纯颜色 → color_palette",
            "单纯情绪 → mood",
            "艺术流派 → artistic_styles"
        ],
        "boundary_rule": "问自己：'这是在描述什么主题/讲什么故事，还是在描述什么颜色/情绪/风格？'如果是主题/故事→visual_theme，否则→对应的专门relation"
    },

    "color_palette": {
        "description": "颜色及配色方案",
        "includes": [
            "单一颜色：red, blue, green, black, white",
            "颜色组合：black_and_white, red_and_blue",
            "配色方案：warm_tones, cool_colors, monochromatic, vibrant_palette"
        ],
        "excludes": [
            "情绪性的'dark' → mood"
        ],
        "boundary_rule": "如果描述颜色或颜色组合→color_palette；如果是情绪词（碰巧有颜色含义）→mood"
    },

    "mood": {
        "description": "情绪、氛围、感受（形容词性）",
        "includes": [
            "情绪形容词：dramatic, romantic, melancholic, tense, joyful, mysterious, dark",
            "氛围描述：ominous, hopeful, suspenseful, serene"
        ],
        "excludes": [
            "叙事主题（romance, war） → visual_theme",
            "颜色（dark作为颜色） → color_palette"
        ],
        "boundary_rule": "如果是形容词AND描述情绪/感受→mood；如果是名词AND是叙事主题→visual_theme"
    },

    "artistic_styles": {
        "description": "艺术流派、风格流派、视觉风格技法",
        "includes": [
            "艺术流派：art_deco, impressionistic, expressionism, cubism",
            "时代风格：vintage, retro, modern, contemporary",
            "电影风格：film_noir, neo_noir, grunge"
        ],
        "excludes": [
            "物理质感（grainy, high_contrast） → texture",
            "视觉概念（minimalism作为概念） → visual_theme"
        ],
        "boundary_rule": "如果描述艺术流派/风格→artistic_styles；如果描述物理质感→texture；如果描述整体概念→visual_theme"
    },

    "texture": {
        "description": "物理/技术层面的视觉质感",
        "includes": [
            "颗粒质感：grainy, film_grain, smooth, rough",
            "对比度：high_contrast, low_contrast, contrasty",
            "焦距效果：soft_focus, sharp, blurred_background"
        ],
        "excludes": [
            "艺术风格 → artistic_styles"
        ],
        "boundary_rule": "如果描述物理/技术参数（颗粒、对比度、焦距）→texture；如果描述艺术风格→artistic_styles"
    },

    "genre": {
        "description": "电影类型",
        "includes": [
            "主流类型：action, romance, thriller, horror, comedy, drama, sci-fi"
        ],
        "excludes": [],
        "boundary_rule": "如果是标准电影类型分类→genre"
    },

    "lighting": {
        "description": "光照条件和打光技术",
        "includes": [
            "光源类型：natural_light, artificial_light, candlelight",
            "打光技术：dramatic_lighting, backlit, front_lit",
            "光照强度：high-key, low-key, bright_lighting"
        ],
        "excludes": [],
        "boundary_rule": ""
    },

    "text_style": {
        "description": "文字排版样式",
        "includes": [
            "字体样式：bold_text, serif_font, sans_serif",
            "排版布局：large_title, minimal_text, centered_text"
        ],
        "excludes": [],
        "boundary_rule": ""
    },

    "symbolism": {
        "description": "具有象征意义的符号和图案",
        "includes": [
            "宗教符号：cross, star_of_david, crescent_moon",
            "国家符号：american_flag, eagle",
            "文化符号：rose, skull, dove"
        ],
        "excludes": [],
        "boundary_rule": ""
    },

    "composition_styles": {
        "description": "构图技法和画面布局方式",
        "includes": [
            "构图法则：rule_of_thirds, golden_ratio, symmetry",
            "取景方式：close-up, wide_shot, medium_shot"
        ],
        "excludes": [],
        "boundary_rule": ""
    },

    "design_element": {
        "description": "图形设计元素",
        "includes": [
            "图形元素：typography, geometric_shapes, border, frame, logo"
        ],
        "excludes": [
            "文字样式 → text_style",
            "构图布局 → composition_styles"
        ],
        "boundary_rule": ""
    },

    "action_behaviors": {
        "description": "物理动作和行为（动词性）",
        "includes": [
            "身体动作：climbing, running, fighting, walking, dancing",
            "行为动作：action_scene, dynamic_movement"
        ],
        "excludes": [],
        "boundary_rule": ""
    },

    "additional_elements": {
        "description": "无法归类到任何标准relation的其他元素",
        "includes": [
            "任何无法归入上述14个relations的entities"
        ],
        "excludes": [],
        "boundary_rule": "只有在确实无法归入任何其他relation时才使用"
    }
}

# 简化版relation列表（供LLM参考）
RELATION_BRIEF = {
    "depicted_subject": "画面主体（人物基础类型、物体、场景）",
    "character_type": "人物角色类型、职业、关系",
    "visual_theme": "叙事主题、概念主题、视觉概念",
    "color_palette": "颜色和配色方案",
    "mood": "情绪、氛围（形容词性）",
    "artistic_styles": "艺术流派、风格流派",
    "texture": "物理/技术质感",
    "genre": "电影类型",
    "lighting": "光照条件和技术",
    "text_style": "文字排版样式",
    "symbolism": "象征符号",
    "composition_styles": "构图技法和布局",
    "design_element": "图形设计元素",
    "action_behaviors": "物理动作和行为",
    "additional_elements": "其他无法归类的元素"
}


def load_entity_data():
    """加载Phase 1数据和relation mapping"""
    # 查找Phase 1数据
    possible_paths = [
        RESULTS_DIR / 'phase1_5percent_exploration.json',
        PROJECT_ROOT / 'outputs' / 'extraction' / 'phase1_results.json',
        PROJECT_ROOT / 'outputs' / 'phase1_results.json',
    ]

    phase1_file = None
    for path in possible_paths:
        if path.exists():
            phase1_file = path
            break

    if not phase1_file:
        raise FileNotFoundError(
            "未找到Phase 1数据文件！请确保以下位置之一存在：\n"
            "  - results/phase1_5percent_exploration.json\n"
            "  - outputs/extraction/phase1_results.json"
        )

    # 加载数据
    print(f"加载Phase 1数据: {phase1_file}")
    with open(phase1_file, 'r') as f:
        phase1_data = json.load(f)

    # 加载relation mapping（兼容v1和v2）
    possible_mapping_files = [
        RESULTS_DIR / 'relation_mapping_final_v2.json',
        RESULTS_DIR / 'relation_mapping_final_v1.json',
        RESULTS_DIR / 'relation_mapping_final.json',
    ]

    mapping_file = None
    for path in possible_mapping_files:
        if path.exists():
            mapping_file = path
            break

    if not mapping_file:
        raise FileNotFoundError(
            "未找到Relation映射文件！请确保以下位置之一存在：\n"
            "  - results/relation_mapping_final_v2.json\n"
            "  - results/relation_mapping_final_v1.json\n"
            "  - results/relation_mapping_final.json"
        )

    print(f"加载Relation映射: {mapping_file}")
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)

    return phase1_data, mapping_data


def extract_entities_by_relation(phase1_data, mapping_data) -> Dict[str, Counter]:
    """提取每个标准relation下的所有entities"""
    relation_mapping = mapping_data['relation_mapping']

    # 提取results列表
    if 'results' in phase1_data:
        phase1_results = phase1_data['results']
    elif isinstance(phase1_data, list):
        phase1_results = phase1_data
    else:
        phase1_results = []

    # 按relation分组entities
    relation_entities = defaultdict(list)

    for movie_result in phase1_results:
        status = movie_result.get('status', movie_result.get('extraction_status', 'success'))
        if status != 'success':
            continue

        for kp in movie_result.get('knowledge_points', []):
            original_relation = kp.get('relation')
            if not original_relation:
                continue

            # 映射到标准relation
            standard_relation = relation_mapping.get(
                original_relation,
                relation_mapping.get(original_relation.lower(), 'additional_elements')
            )

            entity = kp.get('entity')
            if entity:
                relation_entities[standard_relation].append(entity)

    # 统计频率
    relation_entity_counters = {}
    for relation, entities in relation_entities.items():
        relation_entity_counters[relation] = Counter(entities)

    return relation_entity_counters


def build_first_round_prompt(relation: str, entities: List[str], entity_counter: Counter) -> str:
    """构建第一轮prompt：清理和合并entities"""

    rel_def = RELATION_DEFINITIONS.get(relation, {})

    # 其他relations的简短列表（供参考）
    other_relations = "\n".join([
        f"  - {r}: {desc}"
        for r, desc in RELATION_BRIEF.items()
        if r != relation
    ])

    prompt = f"""你是一个专业的知识分类专家。你的任务是清理和优化一个relation下的entity列表。

## Relation定义

**Relation名称**: {relation}

**职责**: {rel_def.get('description', '')}

**包含范围**:
"""

    for item in rel_def.get('includes', []):
        prompt += f"- {item}\n"

    prompt += "\n**排除范围**:\n"
    for item in rel_def.get('excludes', []):
        prompt += f"- {item}\n"

    if rel_def.get('boundary_rule'):
        prompt += f"\n**边界规则**: {rel_def['boundary_rule']}\n"

    prompt += f"""

## 其他Relations（供参考）

{other_relations}

## 当前Entity列表

以下是当前归属于`{relation}`的{len(entities)}个unique entities：

"""

    # 按频率排序显示entities（不显示频率，避免bias）
    sorted_entities = sorted(entities)
    for i, entity in enumerate(sorted_entities, 1):
        prompt += f"{i}. {entity}\n"

    prompt += """

## 你的任务

请对这些entities进行两项操作：

### 1. 剔除不属于该relation的entities

识别哪些entities **不符合**该relation的定义，应该被移除。

对于每个被移除的entity，提供：
- 移除原因（为什么不属于该relation）
- 建议目标relation（应该属于哪个relation）

### 2. 合并同义词/语义相近的entities

识别哪些entities是**同义词**或**语义相近**（表达同一概念的不同表述），应该合并。

对于每个合并组，选择一个**canonical name**（标准名称，优先选择更通用/更标准的表述）。

**重要**：
- 只合并真正的同义词，不要合并语义层级不同的entities（如man vs warrior）
- 只合并同一relation内的entities，不要跨relation合并

## 输出格式

请以JSON格式输出，包含两个部分：

```json
{
  "keep_and_merge": {
    "canonical_name_1": ["entity1", "entity2", "entity3"],
    "canonical_name_2": ["entity4"],
    ...
  },
  "remove": {
    "entity_name": {
      "reason": "移除原因",
      "suggested_relation": "建议目标relation"
    },
    ...
  }
}
```

**说明**：
- `keep_and_merge`: 保留的entities，分组合并。每个group的key是canonical name，value是该组包含的所有entities（包括canonical name自己）
- `remove`: 被移除的entities，key是entity名称，value包含移除原因和建议目标relation

**示例**：
```json
{
  "keep_and_merge": {
    "human_figure": ["human_portrait", "human_figure", "human_figures"],
    "landscape": ["landscape"],
    "building": ["building", "buildings"]
  },
  "remove": {
    "warrior": {
      "reason": "这是角色类型，不是基础人物类型",
      "suggested_relation": "character_type"
    },
    "red": {
      "reason": "这是颜色，不是视觉主体",
      "suggested_relation": "color_palette"
    }
  }
}
```

请仔细分析每个entity，确保分类准确。
"""

    return prompt


def call_gpt4(prompt: str, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    调用GPT-4 API（使用项目现有的MLLM接口）

    Args:
        prompt: 用户prompt
        model: 模型名称
        api_key: API key（可选，如果不提供则从环境变量读取）

    Returns:
        解析后的JSON结果
    """
    from openai import OpenAI

    # 使用api_key参数或环境变量
    client_kwargs = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    # else: OpenAI会自动从环境变量OPENAI_API_KEY读取

    client = OpenAI(**client_kwargs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的知识分类专家，擅长语义分析和实体归类。请严格按照JSON格式输出。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"\n❌ 调用GPT-4失败: {e}")
        print("\n提示：请确保已设置OPENAI_API_KEY环境变量，或使用 --api-key 参数")
        raise


def process_relation_first_round(
    relation: str,
    entity_counter: Counter,
    dry_run: bool = False,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """第一轮处理：清理和合并某个relation的entities"""

    entities = list(entity_counter.keys())

    print(f"\n{'='*80}")
    print(f"第一轮处理: {relation}")
    print(f"{'='*80}")
    print(f"当前entities数量: {len(entities)}")
    print(f"总实例数: {sum(entity_counter.values())}")

    # 构建prompt
    prompt = build_first_round_prompt(relation, entities, entity_counter)

    if dry_run:
        print("\n[DRY RUN] Prompt预览:")
        print(prompt[:1000] + "...\n")
        return {}

    # 调用GPT-4
    print("正在调用GPT-4...")
    result = call_gpt4(prompt, api_key=api_key)

    # 分析结果
    keep_merge = result.get('keep_and_merge', {})
    remove = result.get('remove', {})

    print(f"\n结果:")
    print(f"  保留并合并: {len(keep_merge)} 个groups（合并前：{sum(len(v) for v in keep_merge.values())} entities）")
    print(f"  移除: {len(remove)} entities")

    # 显示合并的groups（如果有合并）
    merged_groups = {k: v for k, v in keep_merge.items() if len(v) > 1}
    if merged_groups:
        print(f"\n  合并的groups:")
        for canonical, members in list(merged_groups.items())[:5]:
            print(f"    - {canonical}: {members}")
        if len(merged_groups) > 5:
            print(f"    ... 还有 {len(merged_groups) - 5} 个groups")

    # 显示被移除的entities
    if remove:
        print(f"\n  被移除的entities:")
        for entity, info in list(remove.items())[:5]:
            print(f"    - {entity} → {info.get('suggested_relation')}")
            print(f"      原因: {info.get('reason')}")
        if len(remove) > 5:
            print(f"    ... 还有 {len(remove) - 5} 个")

    return {
        'relation': relation,
        'original_entity_count': len(entities),
        'keep_and_merge': keep_merge,
        'remove': remove,
        'result_summary': {
            'kept_groups': len(keep_merge),
            'removed_count': len(remove),
            'compression_rate': len(keep_merge) / len(entities) if entities else 0
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2b Entity重分配')
    parser.add_argument('--relation', type=str, help='处理指定的relation')
    parser.add_argument('--all', action='store_true', help='处理所有relations')
    parser.add_argument('--dry-run', action='store_true', help='只生成prompt，不调用API')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key（可选，如不提供则从环境变量OPENAI_API_KEY读取）')
    parser.add_argument('--output', type=str, default='results/entity_redistribution_round1.json',
                       help='输出文件路径')

    args = parser.parse_args()

    # 加载数据
    try:
        phase1_data, mapping_data = load_entity_data()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n提示: 请确保已经运行了Phase 1提取并保存了数据文件。")
        return

    # 提取entities
    relation_entity_counters = extract_entities_by_relation(phase1_data, mapping_data)

    print(f"\n找到 {len(relation_entity_counters)} 个relations:")
    for rel, counter in sorted(relation_entity_counters.items()):
        print(f"  - {rel:25s}: {len(counter):3d} unique entities")

    # 处理relations
    results = {}

    if args.relation:
        # 处理单个relation
        if args.relation not in relation_entity_counters:
            print(f"\n错误: Relation '{args.relation}' 不存在")
            print(f"可用的relations: {list(relation_entity_counters.keys())}")
            return

        result = process_relation_first_round(
            args.relation,
            relation_entity_counters[args.relation],
            dry_run=args.dry_run,
            api_key=args.api_key
        )
        results[args.relation] = result

    elif args.all:
        # 处理所有relations
        for relation, counter in sorted(relation_entity_counters.items()):
            result = process_relation_first_round(
                relation,
                counter,
                dry_run=args.dry_run,
                api_key=args.api_key
            )
            results[relation] = result

    else:
        print("请指定 --relation RELATION_NAME 或 --all")
        return

    # 保存结果
    if not args.dry_run and results:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'metadata': {
                'stage': 'round1_clean_and_merge',
                'relations_processed': list(results.keys()),
                'total_relations': len(results)
            },
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
