#!/usr/bin/env python3
"""
Phase 2b Stage 1: Entity筛选（判断哪些属于该relation）

目标：
1. 对每个relation下的entities，判断哪些真正属于该relation
2. 输出keep列表和remove列表，带validation检查

使用：
  python scripts/phase2b_stage1_filtering.py

输出：
  results/entity_redistribution_stage1_filtering.json
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import sys
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# 导入共享的relation定义和工具函数
from scripts.phase2b_shared import (
    RELATION_DEFINITIONS,
    load_entity_data,
    extract_entities_by_relation,
    call_gpt4
)


def build_single_entity_prompt(relation: str, entity: str) -> str:
    """构建单个entity的判断prompt"""

    rel_def = RELATION_DEFINITIONS.get(relation, {})

    # 构建其他relations的简要说明（供参考）
    other_relations_brief = ""
    for r, r_def in RELATION_DEFINITIONS.items():
        if r == relation:
            continue
        other_relations_brief += f"  - **{r}**: {r_def.get('description', '')}\n"

    prompt = f"""你是一个专业的知识分类专家。你的任务是判断一个entity是否属于指定的relation。

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

## 其他可用的Relations（供参考）

{other_relations_brief}

## 要判断的Entity

**Entity**: `{entity}`

## 你的任务

请判断这个entity **`{entity}`** 是否属于relation **`{relation}`**。

### 判断标准

1. 仔细对照 `{relation}` 的"包含范围"和"排除范围"
2. 应用"边界规则"进行精确判断
3. 如果不属于，判断应该去哪个relation

## 输出格式

请以JSON格式输出：

```json
{{
  "belongs": true/false,
  "reason": "简要说明判断理由",
  "suggested_relation": "如果不属于，建议去哪个relation（如果belongs=true则为null）"
}}
```

**示例1（属于）**：
```json
{{
  "belongs": true,
  "reason": "war是典型的叙事主题，描述战争故事",
  "suggested_relation": null
}}
```

**示例2（不属于）**：
```json
{{
  "belongs": false,
  "reason": "red是单一颜色，不是视觉主题",
  "suggested_relation": "color_palette"
}}
```

请仔细分析这个entity，给出准确判断。
"""

    return prompt


def validate_filtering_result(result: Dict[str, Any], original_entities: List[str]) -> bool:
    """验证Stage 1结果的正确性"""

    keep_entities = result.get('keep', [])
    remove_entities = result.get('remove', {})

    errors = []

    # 检查1: keep和remove是否有重叠
    keep_set = set(keep_entities)
    remove_set = set(remove_entities.keys())
    overlap = keep_set & remove_set

    if overlap:
        errors.append(f"❌ 错误：以下entities同时出现在keep和remove中：{overlap}")

    # 检查2: 是否有entities丢失
    all_processed = keep_set | remove_set
    original_set = set(original_entities)
    missing = original_set - all_processed
    extra = all_processed - original_set

    if missing:
        errors.append(f"⚠️  警告：以下entities未被处理：{missing}")

    if extra:
        errors.append(f"⚠️  警告：出现了原始列表中不存在的entities：{extra}")

    # 打印错误
    if errors:
        print("\n" + "="*80)
        print("⚠️  Validation检查发现问题:")
        print("="*80)
        for error in errors:
            print(error)
        print("="*80)
        return False

    return True


def process_relation_filtering(
    relation: str,
    entity_counter: Counter,
    dry_run: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Stage 1处理：对每个entity单独调用LLM进行筛选"""

    entities = list(entity_counter.keys())

    print(f"\n{'='*80}")
    print(f"Stage 1 - 筛选Relation: {relation}")
    print(f"{'='*80}")
    print(f"原始entities数量: {len(entities)}")
    print(f"总实例数: {sum(entity_counter.values())}")

    if dry_run:
        # Dry-run模式：只显示第一个entity的prompt
        first_entity = entities[0] if entities else "example"
        prompt = build_single_entity_prompt(relation, first_entity)
        print(f"\n[DRY RUN] 示例prompt（entity: {first_entity}）:")
        print(prompt[:1000] + "...\n")
        return {}

    # 逐个处理每个entity
    keep_entities = []
    remove_entities = {}

    print(f"\n开始逐个判断entities（总共{len(entities)}个）...")

    for idx, entity in enumerate(entities, 1):
        # 显示进度
        print(f"\r  处理进度: {idx}/{len(entities)} ({entity})", end="", flush=True)

        # 构建单个entity的prompt
        prompt = build_single_entity_prompt(relation, entity)

        # 调用GPT-4
        try:
            result = call_gpt4(prompt, api_key=api_key, base_url=base_url)

            if result.get('belongs', False):
                keep_entities.append(entity)
            else:
                remove_entities[entity] = {
                    'reason': result.get('reason', '未提供原因'),
                    'suggested_relation': result.get('suggested_relation', 'additional_elements')
                }
        except Exception as e:
            print(f"\n⚠️  处理 '{entity}' 时出错: {e}")
            print(f"    跳过该entity，继续处理下一个...")
            continue

    print("\n")  # 换行

    # Validation检查
    filtering_result = {
        'keep': keep_entities,
        'remove': remove_entities
    }
    is_valid = validate_filtering_result(filtering_result, entities)

    if not is_valid:
        print("\n⚠️  发现validation错误，但仍会保存结果供检查")

    print(f"\nStage 1 结果:")
    print(f"  保留: {len(keep_entities)} entities")
    print(f"  移除: {len(remove_entities)} entities")

    # 显示被移除的entities（前10个）
    if remove_entities:
        print(f"\n  被移除的entities（前10个）:")
        for entity, info in list(remove_entities.items())[:10]:
            print(f"    - {entity} → {info.get('suggested_relation')}")
            print(f"      原因: {info.get('reason')}")
        if len(remove_entities) > 10:
            print(f"    ... 还有 {len(remove_entities) - 10} 个")

    return {
        'relation': relation,
        'original_entity_count': len(entities),
        'keep': keep_entities,
        'remove': remove_entities,
        'validation': {
            'passed': is_valid,
            'kept_count': len(keep_entities),
            'removed_count': len(remove_entities),
            'total_processed': len(keep_entities) + len(remove_entities),
            'expected_total': len(entities)
        }
    }


def main():
    print("="*80)
    print("Phase 2b Stage 1: Entity筛选（判断归属）")
    print("="*80)
    print()

    # 加载数据
    try:
        phase1_data, mapping_data = load_entity_data()
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
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

    # LLM配置
    api_key = None
    base_url = None

    if not dry_run:
        print("\n请配置LLM参数（留空使用默认值）:")
        api_key = input("  API Key (optional, 留空使用环境变量): ").strip() or None
        base_url = input("  Base URL (optional): ").strip() or None

    # 输出文件
    default_output = 'results/entity_redistribution_stage1_filtering.json'
    output = input(f"\n输出文件路径 (default: {default_output}): ").strip() or default_output

    # 显示配置总结
    print(f"\n{'='*80}")
    print("配置总结")
    print(f"{'='*80}")
    print(f"  处理范围: {'所有relations' if process_all else selected_relation}")
    print(f"  Dry-run: {'是' if dry_run else '否'}")
    if not dry_run:
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
        for relation, counter in sorted(relation_entity_counters.items()):
            result = process_relation_filtering(
                relation,
                counter,
                dry_run=dry_run,
                api_key=api_key,
                base_url=base_url
            )
            if result:
                results[relation] = result
    else:
        result = process_relation_filtering(
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
                'stage': 'stage1_filtering',
                'relations_processed': list(results.keys()),
                'total_relations': len(results)
            },
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("✅ Stage 1 完成!")
        print(f"{'='*80}")
        print(f"结果已保存到: {output_path}")
        print("\n下一步:")
        print("  1. 检查Stage 1的筛选结果（keep vs remove）")
        print("  2. 如有必要，手动修正JSON文件")
        print("  3. 运行Stage 2: python scripts/phase2b_stage2_merging.py")


if __name__ == '__main__':
    main()
