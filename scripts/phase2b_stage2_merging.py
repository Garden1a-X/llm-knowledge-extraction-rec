#!/usr/bin/env python3
"""
Phase 2b Stage 2: Entity合并（合并同义词）

目标：
1. 读取Stage 1的筛选结果
2. 对保留的entities进行同义词合并
3. 输出最终的合并结果

使用：
  python scripts/phase2b_stage2_merging.py

输入：
  results/entity_redistribution_stage1_filtering.json (Stage 1输出)

输出：
  results/entity_redistribution_stage2_merged.json
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# 导入共享函数
from scripts.phase2b_shared import RELATION_DEFINITIONS, call_gpt4


def build_merging_prompt(relation: str, entities: List[str]) -> str:
    """构建Stage 2 prompt：仅合并同义词"""

    rel_def = RELATION_DEFINITIONS.get(relation, {})

    prompt = f"""你是一个专业的知识分类专家。你的任务是识别并合并同义词entities。

## 当前Relation

**Relation名称**: {relation}

**职责**: {rel_def.get('description', '')}

## 已确认属于该Relation的Entities

以下是经过筛选后，确认属于`{relation}`的{len(entities)}个entities：

"""

    # 按字母排序显示entities
    sorted_entities = sorted(entities)
    for i, entity in enumerate(sorted_entities, 1):
        prompt += f"{i}. {entity}\n"

    prompt += """

## 你的任务

**请专注于一件事**：识别哪些entities是同义词或语义相近，应该合并。

### 合并标准

- 只合并**真正的同义词**（表达同一概念的不同表述）
  - 例如：`human_portrait`, `human_figure`, `human_figures` → 都是指人物画像
  - 例如：`romance`, `romantic`, `romantic_comedy`, `romantic_drama` → 都围绕浪漫主题

- **不要合并**语义层级不同的entities
  - 例如：`man` 和 `warrior` 虽然都是人物，但层级不同，不应合并
  - 例如：`building` 和 `skyscraper` 虽然相关，但具体程度不同

### 选择Canonical Name

- **必须从输入列表中选择**一个entity作为canonical name
- 优先选择**更通用、更标准**的名称作为group的代表
- 例如：`romance` 优于 `romantic_comedy`
- 例如：`human_figure` 优于 `human_portrait`
- **不要创造新名字**

## 输出格式

请以JSON格式输出：

```json
{
  "canonical_name_1": ["entity1", "entity2", "entity3"],
  "canonical_name_2": ["entity4"],
  ...
}
```

**重要规则**：
- 每个group的key**必须**是输入列表中存在的entity（不能是新创造的名字）
- 每个group的value是该组包含的所有entities（包括canonical name自己）
- 如果某个entity没有同义词，也要包含（value数组只有它自己）
- 所有输入的entities都必须出现在输出中（不能遗漏）

**示例**：
```json
{
  "human_figure": ["human_portrait", "human_figure", "human_figures"],
  "landscape": ["landscape"],
  "romance": ["romance", "romantic", "romantic_comedy"],
  "adventure": ["adventure"]
}
```

请仔细分析，只合并真正的同义词。确保canonical name来自输入列表。
"""

    return prompt


def validate_merging_result(result: Dict[str, List[str]], input_entities: List[str]) -> bool:
    """验证Stage 2结果的正确性"""

    errors = []

    # 检查1: 收集所有输出的entities
    all_output_entities = []
    for canonical, members in result.items():
        all_output_entities.extend(members)

    output_set = set(all_output_entities)
    input_set = set(input_entities)

    # 检查2: 是否有entities丢失
    missing = input_set - output_set
    if missing:
        errors.append(f"❌ 错误：以下entities未出现在输出中：{missing}")

    # 检查3: 是否有重复
    if len(all_output_entities) != len(output_set):
        from collections import Counter
        counts = Counter(all_output_entities)
        duplicates = {e: c for e, c in counts.items() if c > 1}
        errors.append(f"❌ 错误：以下entities重复出现：{duplicates}")

    # 检查4: canonical name是否在输入列表中
    invalid_canonicals = []
    for canonical in result.keys():
        if canonical not in input_set:
            invalid_canonicals.append(canonical)

    if invalid_canonicals:
        errors.append(f"⚠️  警告：以下canonical names不在输入列表中（LLM创造的新名字）：{invalid_canonicals}")

    # 检查5: canonical name是否在自己的members里
    for canonical, members in result.items():
        if canonical not in members:
            errors.append(f"⚠️  警告：canonical name '{canonical}' 不在自己的members列表中")

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


def process_relation_merging(
    relation: str,
    stage1_result: Dict[str, Any],
    dry_run: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Stage 2处理：合并同义词"""

    keep_entities = stage1_result.get('keep', [])

    print(f"\n{'='*80}")
    print(f"Stage 2 - 合并Relation: {relation}")
    print(f"{'='*80}")
    print(f"Stage 1保留的entities数量: {len(keep_entities)}")

    if not keep_entities:
        print("⚠️  没有entities需要合并，跳过")
        return {
            'relation': relation,
            'stage1_kept_count': 0,
            'merged_groups': {},
            'validation': {
                'passed': True,
                'groups_count': 0
            }
        }

    # 构建prompt
    merging_prompt = build_merging_prompt(relation, keep_entities)

    if dry_run:
        print("\n[DRY RUN] Prompt预览:")
        print(merging_prompt[:1000] + "...\n")
        return {}

    # 调用GPT-4
    print("\n正在调用GPT-4进行合并...")
    merging_result = call_gpt4(merging_prompt, api_key=api_key, base_url=base_url)

    # Validation检查
    is_valid = validate_merging_result(merging_result, keep_entities)

    if not is_valid:
        print("\n⚠️  发现validation错误，但仍会保存结果供检查")

    print(f"\nStage 2 结果:")
    print(f"  合并后groups数: {len(merging_result)}")

    # 显示合并的groups（只显示有多个成员的）
    merged_groups = {k: v for k, v in merging_result.items() if len(v) > 1}
    if merged_groups:
        print(f"  其中包含多个成员的groups: {len(merged_groups)}")
        print(f"\n  合并示例（前10个）:")
        for canonical, members in list(merged_groups.items())[:10]:
            print(f"    - {canonical}: {members}")
        if len(merged_groups) > 10:
            print(f"    ... 还有 {len(merged_groups) - 10} 个groups")
    else:
        print(f"  没有需要合并的同义词（每个entity都独立）")

    return {
        'relation': relation,
        'stage1_kept_count': len(keep_entities),
        'merged_groups': merging_result,
        'validation': {
            'passed': is_valid,
            'groups_count': len(merging_result),
            'merged_count': len(merged_groups)
        }
    }


def main():
    print("="*80)
    print("Phase 2b Stage 2: Entity合并（合并同义词）")
    print("="*80)
    print()

    # 默认输入文件
    default_input = 'results/entity_redistribution_stage1_filtering.json'
    input_file = input(f"Stage 1结果文件路径 (default: {default_input}): ").strip() or default_input

    input_path = PROJECT_ROOT / input_file

    # 检查文件是否存在
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在: {input_path}")
        print("\n请先运行 Stage 1: python scripts/phase2b_stage1_filtering.py")
        return

    # 加载Stage 1结果
    print(f"\n加载Stage 1结果: {input_path}")
    with open(input_path, 'r') as f:
        stage1_data = json.load(f)

    stage1_results = stage1_data.get('results', {})

    if not stage1_results:
        print("❌ 错误: Stage 1结果为空")
        return

    print(f"找到 {len(stage1_results)} 个relations的筛选结果:")
    for rel in stage1_results.keys():
        kept = len(stage1_results[rel].get('keep', []))
        print(f"  - {rel:25s}: {kept} entities保留")

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
            for rel in sorted(stage1_results.keys()):
                print(f"  - {rel}")
            continue
        elif relation_input == 'all':
            process_all = True
            selected_relation = None
            break
        elif relation_input in stage1_results:
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
    default_output = 'results/entity_redistribution_stage2_merged.json'
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
        for relation, stage1_result in sorted(stage1_results.items()):
            result = process_relation_merging(
                relation,
                stage1_result,
                dry_run=dry_run,
                api_key=api_key,
                base_url=base_url
            )
            if result:
                results[relation] = result
    else:
        result = process_relation_merging(
            selected_relation,
            stage1_results[selected_relation],
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

        # 合并Stage 1和Stage 2的完整结果
        combined_results = {}
        for relation, stage2_result in results.items():
            stage1_result = stage1_results[relation]
            combined_results[relation] = {
                'relation': relation,
                'original_entity_count': stage1_result.get('original_entity_count', 0),
                'stage1_keep': stage1_result.get('keep', []),
                'stage1_remove': stage1_result.get('remove', {}),
                'stage2_merged_groups': stage2_result.get('merged_groups', {}),
                'summary': {
                    'original_count': stage1_result.get('original_entity_count', 0),
                    'kept_count': len(stage1_result.get('keep', [])),
                    'removed_count': len(stage1_result.get('remove', {})),
                    'final_groups': len(stage2_result.get('merged_groups', {}))
                }
            }

        output_data = {
            'metadata': {
                'stage': 'stage2_merged',
                'relations_processed': list(results.keys()),
                'total_relations': len(results),
                'stage1_input': str(input_path)
            },
            'results': combined_results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("✅ Stage 2 完成!")
        print(f"{'='*80}")
        print(f"结果已保存到: {output_path}")
        print("\n统计:")
        for relation, data in combined_results.items():
            summary = data['summary']
            print(f"  {relation}:")
            print(f"    原始: {summary['original_count']} → 保留: {summary['kept_count']} → 合并为: {summary['final_groups']} groups")
        print("\n下一步:")
        print("  1. 检查Stage 2的合并结果")
        print("  2. 继续处理其他relations或进行entity聚类")


if __name__ == '__main__':
    main()
