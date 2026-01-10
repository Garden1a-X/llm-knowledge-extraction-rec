#!/usr/bin/env python3
"""
使用LLM优化Entity聚类 - 三步迭代法

三步法：
1. Split - 检查并分割语义冲突的clusters
2. Rename - 为每个cluster选择最佳的canonical name
3. Merge - 根据名字合并语义相似的clusters

可以多轮迭代，每轮跑完检查结果，不满意可以继续跑

用法:
    python scripts/refine_entity_mapping.py [--continue-from FILE] [--start-step N]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.entity_refiner import EntityRefiner


def main():
    parser = argparse.ArgumentParser(description='Entity聚类LLM优化 - 三步迭代法')
    parser.add_argument('--continue-from', type=str, help='从中间结果继续（如results/step1_split.json）')
    parser.add_argument('--start-step', type=int, default=1, choices=[1, 2, 3],
                       help='从第几步开始: 1=split, 2=rename, 3=merge')
    args = parser.parse_args()

    print("="*80)
    print("Entity聚类LLM优化工具 - 三步迭代法")
    print("="*80)
    print()
    print("工作流程:")
    print("  第一步: Split - 分割语义冲突的clusters")
    print("  第二步: Rename - 选择最佳canonical names")
    print("  第三步: Merge - 合并语义相似的clusters")
    print()

    # 配置
    if args.continue_from:
        mapping_file = args.continue_from
        print(f"📥 继续处理: {mapping_file}")
        print(f"   从第 {args.start_step} 步开始")
    else:
        mapping_file = 'results/entity_mapping_bertopic_optimized.json'
        print(f"📥 初始输入: {mapping_file}")

    output_path = 'results/entity_mapping_refined.json'

    # 检查输入文件
    if not Path(mapping_file).exists():
        print(f"❌ 错误: 找不到entity mapping文件: {mapping_file}")
        if not args.continue_from:
            print("请先运行Phase 2b Stage 2 (entity clustering)")
        return

    # 初始化refiner
    refiner = EntityRefiner(mapping_file, output_dir='results')

    # 加载mappings
    refiner.load_mappings(from_file=args.continue_from if args.continue_from else None)

    # LLM配置
    print(f"\n{'='*80}")
    print("LLM配置")
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
    if args.start_step > 1:
        print(f"  开始步骤: {args.start_step}")

    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 执行三步优化
    try:
        refined_mappings = refiner.refine_iterative(
            backend=backend,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            start_from_step=args.start_step
        )

        # 保存最终结果
        refiner.save_final_results(output_path)

        print(f"\n{'='*80}")
        print("✅ 优化完成!")
        print(f"{'='*80}")
        print()
        print("中间结果:")
        print("  results/step1_split.json - 第一步：分割语义冲突")
        print("  results/step2_renamed.json - 第二步：重命名clusters")
        print("  results/step3_merged.json - 第三步：合并相似clusters")
        print()
        print(f"最终结果: {output_path}")
        print()
        print("下一步:")
        print("  1. 检查最终结果是否满意")
        print("  2. 如果不满意，可以继续优化:")
        print("     python scripts/refine_entity_mapping.py --continue-from results/step3_merged.json")
        print("  3. 如果满意，可以用于后续的知识图谱构建")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
