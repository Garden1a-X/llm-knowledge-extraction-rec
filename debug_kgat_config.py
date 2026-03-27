#!/usr/bin/env python3
"""
调试脚本：检查KGAT实际加载的配置

重点检查load_col字段，看看link是否被正确加载
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from recbole.config import Config
from recbole.data import create_dataset
import yaml

print("="*70)
print("DEBUG: KGAT配置加载")
print("="*70)

# 1. 加载YAML配置文件
config_file = 'configs/recbole_kgat.yaml'
print(f"\n1. 读取配置文件: {config_file}")
with open(config_file, 'r') as f:
    yaml_config = yaml.safe_load(f)

print(f"   YAML中的load_col:")
for key, value in yaml_config.get('load_col', {}).items():
    print(f"     {key}: {value}")

# 2. 通过RecBole加载配置
print(f"\n2. RecBole加载配置...")
config = Config(
    model='KGAT',
    dataset='ml-1m',
    config_file_list=[config_file]
)

print(f"   RecBole config['load_col'] 类型: {type(config['load_col'])}")
print(f"   RecBole config['load_col'] 值: {config['load_col']}")

if config['load_col'] is not None:
    print(f"   load_col字典的keys: {list(config['load_col'].keys())}")
    print(f"   'link' in load_col: {'link' in config['load_col']}")

    if 'link' in config['load_col']:
        print(f"   load_col['link']: {config['load_col']['link']}")

# 3. 尝试创建dataset
print(f"\n3. 尝试创建dataset...")
try:
    dataset = create_dataset(config)
    print(f"   ✓ Dataset创建成功!")
    print(f"   Dataset类型: {type(dataset)}")

    # 检查是否有link相关属性
    if hasattr(dataset, 'link_df'):
        print(f"   dataset.link_df: {dataset.link_df.shape if dataset.link_df is not None else None}")

except Exception as e:
    print(f"   ✗ Dataset创建失败!")
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
