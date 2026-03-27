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
import json
from pathlib import Path
from datetime import datetime

# Add RecBole to path if needed
try:
    from recbole.quick_start import run_recbole
except ImportError:
    print("Error: RecBole not found. Please install RecBole first.")
    sys.exit(1)


class KGAblationRunner:
    def __init__(self):
        # 数据路径 - 注意：应该是基础路径，RecBole会自动找ml-1m子目录
        self.visual_kg_dir = "/data/xuao/llm-knowledge-extraction-rec/data/recbole"
        self.metadata_kg_dir = "/data/xuao/KG4RecEval/dataset"

        # 输出目录
        project_root = Path(__file__).parent.parent
        self.output_dir = project_root / "outputs" / "kg_ablation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def get_model_config(self, method):
        """获取模型特定的配置参数"""
        # 通用配置
        base_config = {
            # Data
            'dataset': 'ml-1m',
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                'kg': ['head_id', 'relation_id', 'tail_id'],
                'link': ['item_id', 'entity_id']
            },

            # Data split (same as ours)
            'eval_args': {
                'split': {'RS': [0.7, 0.1, 0.2]},
                'order': 'TO',
                'group_by': 'user',
                'mode': 'uni100'
            },

            # Evaluation metrics
            'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
            'topk': [5, 10, 20],
            'valid_metric': 'NDCG@10',

            # Training
            'epochs': 300,
            'train_batch_size': 2048,
            'learning_rate': 0.001,
            'stopping_step': 10,

            # Random seed
            'seed': 42,

            # Reproducibility
            'reproducibility': True,
            'state': 'INFO',
            'show_progress': True,
        }

        # 模型特定参数
        if method == "KGAT":
            model_config = {
                'embedding_size': 64,
                'kg_embedding_size': 64,
                'reg_weight': 0.0001,
                'aggregator_type': 'bi-interaction',
                'n_layers': 2,
                'mess_dropout': 0.1,
            }
        elif method == "KGCN":
            model_config = {
                'embedding_size': 64,
                'kg_embedding_size': 64,
                'reg_weight': 0.0001,
                'neighbor_sample_size': 8,
                'n_iter': 1,
                'aggregator': 'sum',
            }
        elif method == "KGIN":
            model_config = {
                'embedding_size': 64,
                'kg_embedding_size': 64,
                'reg_weight': 0.0001,
                'n_layers': 3,
                'context_hops': 3,
                'node_dropout': 0.1,
                'mess_dropout': 0.1,
                'ind': 'distance',
            }
        else:
            model_config = {}

        # 合并配置
        base_config.update(model_config)
        return base_config

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

        # 获取模型配置
        config_dict = self.get_model_config(method)

        # 添加路径配置
        config_dict['data_path'] = data_path
        config_dict['checkpoint_dir'] = str(exp_output_dir / 'checkpoints')

        print(f"Data path: {data_path}")
        print(f"Output directory: {exp_output_dir}")
        print(f"Evaluation mode: uni100")
        print()

        # 运行实验
        try:
            result = run_recbole(
                model=method,
                dataset='ml-1m',
                config_dict=config_dict,
                saved=True
            )

            # 解析结果
            if isinstance(result, tuple) and len(result) == 2:
                best_valid_score, test_result = result
                result_dict = {
                    'best_valid_score': float(best_valid_score) if best_valid_score is not None else None,
                    'test_result': test_result
                }
            elif isinstance(result, dict):
                if 'test_result' not in result:
                    result_dict = {'test_result': result}
                else:
                    result_dict = result
            else:
                result_dict = {'test_result': result}

            # 保存结果
            results_path = exp_output_dir / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)

            print(f"\n✅ {method} + {kg_type} KG completed successfully")
            print(f"Results saved to: {results_path}")

            # 返回标准化的结果
            test_result = result_dict.get('test_result', {})
            if isinstance(test_result, dict):
                normalized = {}
                for key, value in test_result.items():
                    normalized[key.lower()] = value
                return normalized

            return None

        except Exception as e:
            print(f"\n❌ {method} + {kg_type} KG failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
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

            exp_results = self.run_experiment(method, kg_type)

            if exp_results:
                if method not in results:
                    results[method] = {}
                results[method][kg_type] = exp_results
            else:
                print(f"⚠️  Experiment failed or could not parse results")

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
    visual_data_dir = Path(runner.visual_kg_dir) / "ml-1m"
    metadata_data_dir = Path(runner.metadata_kg_dir) / "ml-1m"

    if not visual_data_dir.exists():
        print(f"❌ Visual KG directory not found: {visual_data_dir}")
        return

    if not metadata_data_dir.exists():
        print(f"❌ Metadata KG directory not found: {metadata_data_dir}")
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
