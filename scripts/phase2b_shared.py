"""
Phase 2b 共享模块：包含RELATION_DEFINITIONS和共享工具函数
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

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
        "description": "整体视觉/叙事主题、概念性主题（不包括电影类型）",
        "includes": [
            "叙事主题：war, romance, adventure, survival, coming_of_age, betrayal, revenge",
            "概念主题：urban_life, nature, technology, isolation, freedom, tradition",
            "视觉概念：minimalism, surrealism, realism, abstraction, symmetry_theme"
        ],
        "excludes": [
            "单纯颜色 → color_palette",
            "单纯情绪 → mood",
            "艺术流派 → artistic_styles",
            "电影类型（horror, thriller, comedy, action, sci-fi等） → genre"
        ],
        "boundary_rule": "问自己：'这是在描述什么主题/讲什么故事的内容，还是什么类型的电影/什么颜色/情绪/风格？'主题例子：war战争、freedom自由、betrayal背叛。类型例子：horror恐怖片、thriller惊悚片→genre"
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

    # 合并depicted_entities到depicted_subject（根据Phase 2b设计决策）
    if 'depicted_entities' in relation_entities:
        print("⚠️  检测到depicted_entities，正在合并到depicted_subject...")
        if 'depicted_subject' not in relation_entities:
            relation_entities['depicted_subject'] = []
        relation_entities['depicted_subject'].extend(relation_entities['depicted_entities'])
        del relation_entities['depicted_entities']
        print(f"   ✓ 已合并 depicted_entities → depicted_subject")

    # 统计频率
    relation_entity_counters = {}
    for relation, entities in relation_entities.items():
        relation_entity_counters[relation] = Counter(entities)

    return relation_entity_counters


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
