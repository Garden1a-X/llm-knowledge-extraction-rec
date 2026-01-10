#!/usr/bin/env python3
"""
使用LLM修正Entity聚类的语义冲突

检测并分离BERTopic产生的语义不一致clusters（如protagonist + antagonist）

用法:
    python scripts/refine_entity_mapping.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.entity_refiner import EntityRefiner


def main():
    print("="*80)
    print("Entity聚类LLM语义修正工具")
    print("="*80)
    print()

    # 配置
    mapping_file = 'results/entity_mapping_bertopic_optimized.json'
    output_path = 'results/entity_mapping_refined.json'

    # 检查输入文件
    if not Path(mapping_file).exists():
        print(f"❌ 错误: 找不到entity mapping文件: {mapping_file}")
        print("请先运行Phase 2b Stage 2 (entity clustering)")
        return

    # 初始化refiner
    refiner = EntityRefiner(mapping_file)

    # 加载mappings
    refiner.load_mappings()

    # LLM配置
    print(f"\n{'='*80}")
    print("准备使用LLM进行语义验证")
    print(f"{'='*80}")
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
    print(f"  输入: {mapping_file}")
    print(f"  输出: {output_path}")

    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 执行LLM检查和修正
    try:
        refined_mappings = refiner.refine_all(
            backend=backend,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )

        # 保存结果
        refiner.save_results(
            refined_mappings=refined_mappings,
            output_path=output_path
        )

        print(f"\n{'='*80}")
        print("✅ 语义修正完成!")
        print(f"{'='*80}")
        print()
        print("下一步:")
        print("  1. 检查 entity_mapping_refined.json 结果是否合理")
        print("  2. 使用修正后的mappings进行知识图谱构建")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
