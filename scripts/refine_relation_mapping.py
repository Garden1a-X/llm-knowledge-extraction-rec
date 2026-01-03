#!/usr/bin/env python3
"""
使用LLM微调Relation映射 - 修正embedding聚类的语义问题

用法:
    python scripts/refine_relation_mapping.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.relation_refiner import RelationRefiner


def main():
    print("="*80)
    print("Relation映射LLM微调工具")
    print("="*80)
    print()

    # 配置
    base_mapping_path = 'results/relation_mapping_final.json'
    output_path = 'results/relation_mapping_final_v2.json'

    # 检查输入文件
    if not Path(base_mapping_path).exists():
        print(f"❌ 错误: 找不到基础映射文件: {base_mapping_path}")
        print("请先运行relation聚类脚本生成基础映射")
        return

    # 初始化refiner
    refiner = RelationRefiner(base_mapping_path)

    # 加载基础映射
    refiner.load_base_mapping()

    # 分析问题
    print(f"\n{'='*80}")
    print("分析当前映射的问题...")
    print(f"{'='*80}")
    issues = refiner.analyze_mapping_issues()

    # 询问用户是否继续
    print(f"\n{'='*80}")
    print("准备使用LLM进行微调")
    print(f"{'='*80}")

    # LLM配置
    print("\n请配置LLM参数（留空使用默认值）:")
    backend = input("  Backend (default: openai): ").strip() or "openai"
    model = input("  Model (default: gpt-4o-mini): ").strip() or "gpt-4o-mini"
    api_key = input("  API Key (optional, 留空使用环境变量): ").strip() or None
    base_url = input("  Base URL (optional): ").strip() or None
    temperature = input("  Temperature (default: 0.0): ").strip()
    temperature = float(temperature) if temperature else 0.0

    # 确认
    print(f"\n配置:")
    print(f"  Backend: {backend}")
    print(f"  Model: {model}")
    print(f"  Temperature: {temperature}")
    print(f"  输入: {base_mapping_path}")
    print(f"  输出: {output_path}")

    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 执行LLM微调
    try:
        refined_data = refiner.refine_with_llm(
            backend=backend,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            issues=issues
        )

        # 保存结果
        refiner.save_refined_mapping(output_path, refined_data)

        # 显示对比
        print(f"\n{'='*80}")
        print("微调前后对比")
        print(f"{'='*80}")
        print(f"标准relations数: "
              f"{refiner.base_mapping['metadata']['standard_relations_after']} → "
              f"{refined_data['summary']['relations_after']}")
        print(f"总改动数: {refined_data['summary']['total_changes']}")

        print(f"\n主要改进:")
        improvements = refined_data['summary'].get('major_improvements',
                                                     refined_data['summary'].get('major_changes', []))
        for improvement in improvements:
            print(f"  - {improvement}")

        print(f"\n{'='*80}")
        print("✅ 微调完成!")
        print(f"{'='*80}")
        print(f"精炼后的映射已保存到: {output_path}")
        print()
        print("下一步:")
        print("  1. 检查映射结果是否合理")
        print("  2. 如果满意，将 relation_mapping_final_v2.json 重命名为 relation_mapping_final_v1.json")
        print("  3. 继续进行 Phase 2b Entity聚类")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
