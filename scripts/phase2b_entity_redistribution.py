#!/usr/bin/env python3
"""
Phase 2b-0: Entity重分配 - 第一轮清理与合并

目标：
1. 按relation清理entities（剔除不属于的 + 合并同义词）
2. 为每个relation生成清理后的entity列表

使用：
  python scripts/phase2b_entity_redistribution.py

然后按照交互式提示输入配置：
  - 选择要处理的relation（或all）
  - Dry-run模式（可选）
  - LLM配置（API key, base URL等）
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import os
import sys
from typing import Dict, List, Tuple, Any, Optional

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

    # 构建其他relations的完整定义（供精确参考）
    other_relations_detail = ""
    for r, r_def in RELATION_DEFINITIONS.items():
        if r == relation:
            continue
        other_relations_detail += f"\n### {r}\n"
        other_relations_detail += f"**职责**: {r_def.get('description', '')}\n\n"
        other_relations_detail += "**包含范围**:\n"
        for item in r_def.get('includes', []):
            other_relations_detail += f"- {item}\n"
        other_relations_detail += "\n**排除范围**:\n"
        for item in r_def.get('excludes', []):
            other_relations_detail += f"- {item}\n"
        if r_def.get('boundary_rule'):
            other_relations_detail += f"\n**边界规则**: {r_def['boundary_rule']}\n"
        other_relations_detail += "\n"

    prompt = f"""你是一个专业的知识分类专家。你的任务是清理和优化一个relation下的entity列表。

## 当前处理的Relation

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

## 所有其他Relations的完整定义（供精确判断移除目标）

{other_relations_detail}

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

**非常重要**：严格根据relation的定义和边界规则，识别哪些entities **不符合**该relation的定义。

判断标准：
- 仔细对照该relation的"包含范围"和"排除范围"
- 应用"边界规则"进行精确判断
- 参考其他relations的完整定义，找到更合适的目标relation

对于每个被移除的entity，提供：
- 移除原因（明确说明为什么不属于当前relation）
- 建议目标relation（根据其他relations的定义，判断应该属于哪个relation）

### 2. 合并同义词/语义相近的entities

识别哪些entities是**同义词**或**语义相近**（表达同一概念的不同表述），应该合并。

对于每个合并组，选择一个**canonical name**（标准名称，优先选择更通用/更标准的表述）。

**重要**：
- 只合并真正的同义词，不要合并语义层级不同的entities（如man vs warrior）
- 只合并同一relation内应该保留的entities，不要合并应该被移除的entities

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
      "reason": "移除原因（说明为什么不符合当前relation定义）",
      "suggested_relation": "建议目标relation（根据其他relations定义判断）"
    },
    ...
  }
}
```

**说明**：
- `keep_and_merge`: 保留在当前relation的entities，分组合并。每个group的key是canonical name，value是该组包含的所有entities（包括canonical name自己）
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


def call_gpt4(
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    调用GPT-4 API（采用与mllm_interface相同的方式）

    Args:
        prompt: 用户prompt
        model: 模型名称
        api_key: API key（可选，如果不提供则从环境变量OPENAI_API_KEY读取）
        base_url: 可选的base URL（用于OpenAI兼容的API，如本地vLLM）

    Returns:
        解析后的JSON结果
    """
    from openai import OpenAI

    # 使用api_key和base_url参数，或从环境变量读取
    client_kwargs = {}
    if api_key is not None:
        client_kwargs['api_key'] = api_key
    # else: OpenAI会自动从环境变量OPENAI_API_KEY读取

    if base_url:
        client_kwargs['base_url'] = base_url

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
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
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
    result = call_gpt4(prompt, api_key=api_key, base_url=base_url)

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
    print("="*80)
    print("Phase 2b Entity重分配 - 第一轮清理与合并")
    print("="*80)
    print()

    # 加载数据
    try:
        phase1_data, mapping_data = load_entity_data()
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("\n提示: 请确保已经运行了Phase 1提取并保存了数据文件。")
        return

    # 提取entities
    relation_entity_counters = extract_entities_by_relation(phase1_data, mapping_data)

    print(f"\n找到 {len(relation_entity_counters)} 个relations:")
    for rel, counter in sorted(relation_entity_counters.items()):
        print(f"  - {rel:25s}: {len(counter):3d} unique entities")

    # 交互式配置
    print(f"\n{'='*80}")
    print("配置处理参数")
    print(f"{'='*80}")

    # 选择要处理的relation
    print("\n请选择要处理的relation:")
    print("  - 输入relation名称（如 visual_theme）")
    print("  - 输入 'all' 处理所有relations")
    print("  - 输入 'list' 查看所有可用的relations")

    while True:
        relation_input = input("\nRelation: ").strip()
        if relation_input == 'list':
            print("\n可用的relations:")
            for rel in sorted(relation_entity_counters.keys()):
                print(f"  - {rel}")
            continue
        elif relation_input == 'all':
            process_all = True
            selected_relation = None
            break
        elif relation_input in relation_entity_counters:
            process_all = False
            selected_relation = relation_input
            break
        else:
            print(f"❌ 错误: Relation '{relation_input}' 不存在，请重新输入")

    # Dry-run选项
    dry_run_input = input("\nDry-run模式（只看prompt不调用API）? (y/n, default: n): ").strip().lower()
    dry_run = dry_run_input == 'y'

    # LLM配置（仅在非dry-run模式下询问）
    api_key = None
    base_url = None
    model = "gpt-4o-mini"
    temperature = 0.1

    if not dry_run:
        print("\n请配置LLM参数（留空使用默认值）:")
        model = input("  Model (default: gpt-4o-mini): ").strip() or "gpt-4o-mini"
        api_key = input("  API Key (optional, 留空使用环境变量): ").strip() or None
        base_url = input("  Base URL (optional): ").strip() or None
        temp_input = input("  Temperature (default: 0.1): ").strip()
        temperature = float(temp_input) if temp_input else 0.1

    # 输出文件
    default_output = 'results/entity_redistribution_round1.json'
    output = input(f"\n输出文件路径 (default: {default_output}): ").strip() or default_output

    # 显示配置总结
    print(f"\n{'='*80}")
    print("配置总结")
    print(f"{'='*80}")
    print(f"  处理范围: {'所有relations' if process_all else selected_relation}")
    print(f"  Dry-run: {'是' if dry_run else '否'}")
    if not dry_run:
        print(f"  Model: {model}")
        print(f"  Temperature: {temperature}")
        print(f"  API Key: {'已设置' if api_key else '使用环境变量'}")
        print(f"  Base URL: {base_url if base_url else '默认'}")
    print(f"  输出文件: {output}")

    # 确认
    if not dry_run:
        confirm = input("\n是否继续? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return

    # 处理relations
    results = {}

    if process_all:
        # 处理所有relations
        for relation, counter in sorted(relation_entity_counters.items()):
            result = process_relation_first_round(
                relation,
                counter,
                dry_run=dry_run,
                api_key=api_key,
                base_url=base_url
            )
            if result:  # dry-run会返回空dict
                results[relation] = result
    else:
        # 处理单个relation
        result = process_relation_first_round(
            selected_relation,
            relation_entity_counters[selected_relation],
            dry_run=dry_run,
            api_key=api_key,
            base_url=base_url
        )
        if result:
            results[selected_relation] = result

    # 保存结果
    if not dry_run and results:
        output_path = PROJECT_ROOT / output
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

        print(f"\n{'='*80}")
        print("✅ 处理完成!")
        print(f"{'='*80}")
        print(f"结果已保存到: {output_path}")
        print("\n下一步:")
        print("  1. 检查每个relation的清理结果是否合理")
        print("  2. 如果满意，继续进行第二轮孤儿entity重分配")
        print("  3. 最后进行entity聚类")


if __name__ == '__main__':
    main()
