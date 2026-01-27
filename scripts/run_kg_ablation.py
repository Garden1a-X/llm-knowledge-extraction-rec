#!/usr/bin/env python3
"""
图谱有效性消融实验脚本

对比不同KG推荐方法在两种图谱上的性能：
- Metadata-based KG: 从数据集原始metadata构建（genre, year等）
- Visual KG (Ours): 用LLM从海报提取的视觉知识图谱

Methods: KGAT, KGCN, KGIN
Total experiments: 5 (KGAT+Metadata already done, need to run 5 more)
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class KGAblationRunner:
    def __init__(self):
        # 数据路径
        self.visual_kg_dir = "/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m"
        self.metadata_kg_dir = "/data/xuao/KG4RecEval/dataset/ml-1m"

        # 输出目录
        self.output_dir = project_root / "outputs" / "kg_ablation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # run_baseline.py 脚本路径
        self.run_baseline_script = project_root / "baselines" / "run_baseline.py"

        # 实验配置
        self.methods = ["KGAT", "KGCN", "KGIN"]

        # KGAT + Metadata 已完成，NDCG@10 = 0.2222
        self.completed = {
            ("KGAT", "metadata"): {
                "ndcg@10": 0.2222,
                "recall@10": 0.1518
            }
        }

        # 实验列表（需要运行的）
        self.experiments = [
            # KGAT with Visual KG (metadata already done)
            {"method": "KGAT", "kg_type": "visual"},

            # KGCN with both KGs
            {"method": "KGCN", "kg_type": "metadata"},
            {"method": "KGCN", "kg_type": "visual"},

            # KGIN with both KGs
            {"method": "KGIN", "kg_type": "metadata"},
            {"method": "KGIN", "kg_type": "visual"},
        ]

        print(f"\n{'='*80}")
        print(f"KG Ablation Study Setup")
        print(f"{'='*80}\n")
        print(f"Methods: {', '.join(self.methods)}")
        print(f"Visual KG dir: {self.visual_kg_dir}")
        print(f"Metadata KG dir: {self.metadata_kg_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"\nTotal experiments to run: {len(self.experiments)}")
        print(f"Already completed: {len(self.completed)}")
        print(f"{'='*80}\n")

    def run_experiment(self, method, kg_type):
        """运行单个实验"""
        print(f"\n{'='*80}")
        print(f"Running {method} with {kg_type} KG")
        print(f"{'='*80}\n")

        # 选择数据路径
        if kg_type == "visual":
            data_path = self.visual_kg_dir
        else:
            data_path = self.metadata_kg_dir

        # 输出目录
        exp_output_dir = self.output_dir / f"{method}_{kg_type}"
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        # 构建命令 - 仿照KGAT的跑法
        cmd = [
            "python",
            str(self.run_baseline_script),
            "--model", method,
            "--dataset", "ml-1m",
            "--data_path", data_path,
            "--output_dir", str(exp_output_dir),
            "--device", "cuda",
            "--use_kg",  # 所有KG方法都需要这个flag
            "--seed", "42",  # 只跑1个trial，用seed=42
            "--epochs", "300"
        ]

        print(f"Command: {' '.join(cmd)}\n")

        # 运行实验
        log_file = exp_output_dir / "run.log"
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False
                )

            if result.returncode == 0:
                print(f"✅ {method} + {kg_type} KG completed successfully")
                return True
            else:
                print(f"❌ {method} + {kg_type} KG failed with return code {result.returncode}")
                print(f"   Check log: {log_file}")
                return False

        except Exception as e:
            print(f"❌ Error running {method} + {kg_type} KG: {e}")
            return False

    def parse_results(self, method, kg_type):
        """解析实验结果"""
        exp_output_dir = self.output_dir / f"{method}_{kg_type}"

        # 查找最新的results.json文件
        result_files = list(exp_output_dir.glob("*/results.json"))

        if not result_files:
            return None

        # 使用最新的结果文件
        latest_result_file = sorted(result_files, key=lambda x: x.stat().st_mtime)[-1]

        try:
            with open(latest_result_file, 'r') as f:
                data = json.load(f)

            # 提取test_result
            test_result = data.get('test_result', {})

            if test_result:
                # 标准化指标名称（小写）
                normalized = {}
                for key, value in test_result.items():
                    normalized[key.lower()] = value

                return normalized

        except Exception as e:
            print(f"Warning: Failed to parse results for {method} + {kg_type}: {e}")

        return None

    def run_all(self):
        """运行所有实验"""
        results = {}

        # 添加已完成的实验
        for (method, kg_type), res in self.completed.items():
            if method not in results:
                results[method] = {}
            results[method][kg_type] = res

        # 运行新实验
        for i, exp in enumerate(self.experiments, 1):
            method = exp["method"]
            kg_type = exp["kg_type"]

            print(f"\n{'='*80}")
            print(f"Experiment [{i}/{len(self.experiments)}]")
            print(f"{'='*80}")

            success = self.run_experiment(method, kg_type)

            if success:
                # 解析结果
                exp_results = self.parse_results(method, kg_type)

                if exp_results:
                    if method not in results:
                        results[method] = {}
                    results[method][kg_type] = exp_results
                else:
                    print(f"⚠️  Could not parse results for {method} + {kg_type}")

        # 保存汇总结果
        summary_file = self.output_dir / "kg_ablation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"All experiments completed!")
        print(f"Results saved to: {summary_file}")
        print(f"{'='*80}\n")

        # 打印结果对比
        self.print_comparison(results)

        return results

    def print_comparison(self, results):
        """打印结果对比表格"""
        print("\n" + "="*100)
        print("KG Ablation Study Results: Visual KG vs Metadata-based KG")
        print("="*100 + "\n")

        # 表头
        print(f"{'Method':<12} | {'KG Type':<12} | {'NDCG@10':<10} | {'Recall@10':<10} | {'Improvement':<12}")
        print("-" * 100)

        for method in self.methods:
            if method not in results:
                print(f"{method:<12} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12}")
                continue

            meta_res = results[method].get("metadata", {})
            visual_res = results[method].get("visual", {})

            # Metadata KG
            if meta_res:
                ndcg = meta_res.get("ndcg@10", 0)
                recall = meta_res.get("recall@10", 0)
                print(f"{method:<12} | {'Metadata':<12} | {ndcg:<10.4f} | {recall:<10.4f} | {'-':<12}")

            # Visual KG
            if visual_res:
                ndcg = visual_res.get("ndcg@10", 0)
                recall = visual_res.get("recall@10", 0)

                # 计算提升
                if meta_res and meta_res.get("ndcg@10", 0) > 0:
                    meta_ndcg = meta_res.get("ndcg@10", 0)
                    improvement = (ndcg - meta_ndcg) / meta_ndcg * 100
                    improvement_str = f"{improvement:+.2f}%"
                else:
                    improvement_str = "N/A"

                print(f"{method:<12} | {'Visual (Ours)':<12} | {ndcg:<10.4f} | {recall:<10.4f} | {improvement_str:<12}")

            print("-" * 100)

        print("\n" + "="*100)
        print("\nKey Finding:")
        print("  - If improvement > 0: Our Visual KG is better than Metadata-based KG")
        print("  - This validates that LLM-extracted visual knowledge provides more value")
        print("="*100 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run KG ablation study')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print experiments without running them')

    args = parser.parse_args()

    runner = KGAblationRunner()

    # 检查数据目录是否存在
    if not Path(runner.visual_kg_dir).exists():
        print(f"❌ Visual KG directory not found: {runner.visual_kg_dir}")
        return

    if not Path(runner.metadata_kg_dir).exists():
        print(f"❌ Metadata KG directory not found: {runner.metadata_kg_dir}")
        return

    if args.dry_run:
        print("\n🔍 DRY RUN - Experiments to run:")
        for i, exp in enumerate(runner.experiments, 1):
            print(f"  [{i}] {exp['method']} + {exp['kg_type']} KG")
        print(f"\nTotal: {len(runner.experiments)} experiments\n")
        return

    # 运行所有实验
    runner.run_all()


if __name__ == "__main__":
    main()
