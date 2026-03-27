#!/usr/bin/env python3
"""
直接使用RecBole API运行KGAT，跳过run_baseline.py的配置覆盖问题
"""

from recbole.quick_start import run_recbole

# 直接使用配置文件运行
result = run_recbole(
    model='KGAT',
    dataset='ml-1m',
    config_file_list=['configs/recbole_kgat.yaml'],
    saved=True
)

print("\n" + "="*80)
print("KGAT Training完成!")
print("="*80)
print(f"Best validation score: {result['best_valid_score']}")
print(f"\nTest results:")
for metric, value in result['test_result'].items():
    print(f"  {metric}: {value:.4f}")
print("="*80)
