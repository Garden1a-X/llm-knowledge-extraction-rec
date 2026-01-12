#!/usr/bin/env python3
"""
Configuration management

使用YAML文件管理实验配置。
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据相关配置"""
    item_kg_path: str = 'data/recbole/ml-1m/ml-1m.item.kg'
    user_kg_path: str = 'data/recbole/ml-1m/ml-1m.user.kg'
    inter_path: str = 'data/recbole/ml-1m/ml-1m.inter'
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    time_based_split: bool = True
    per_user_split: bool = True  # Per-user Random Split（对齐RecBole RS）
    min_rating: float = 4.0


@dataclass
class ModelConfig:
    """模型相关配置"""
    embedding_dim: int = 64
    num_gnn_layers: int = 2
    gat_heads: int = 4
    dropout: float = 0.2
    use_mask: bool = True
    mask_min_freq: int = 5
    mask_max_freq: int = 1000


@dataclass
class LossConfig:
    """损失函数相关配置"""
    # 损失权重
    alpha_contrast: float = 0.1   # 多视图对比
    beta_align: float = 0.05       # Entity-Item对齐
    gamma_mask: float = 0.01       # Mask正则

    # InfoNCE temperature
    temperature_rec: float = 0.2
    temperature_contrast: float = 0.1

    # Mask正则参数
    lambda_sparse: float = 1.0
    lambda_entropy: float = 0.1

    # 对齐损失负采样
    num_neg_align: int = 5


@dataclass
class TrainConfig:
    """训练相关配置"""
    batch_size: int = 1024
    num_negatives: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 300
    early_stop_patience: int = 20
    eval_every: int = 5
    log_every: int = 1
    save_every: int = 10
    num_workers: int = 4
    device: str = 'cuda'
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    name: str = 'ours_full'
    description: str = 'Full model with all components'

    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # 消融实验标志
    ablation: Dict[str, bool] = field(default_factory=lambda: {
        'use_contrast': True,
        'use_align': True,
        'use_mask': True,
        'use_cf_view': True,
        'use_kg_view': True,
    })

    # 输出路径
    output_dir: str = 'outputs'
    checkpoint_dir: str = 'checkpoints'


def load_config(config_path: str) -> ExperimentConfig:
    """
    从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        config: ExperimentConfig对象
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 创建配置对象
    config = ExperimentConfig()

    # 更新配置
    if 'name' in config_dict:
        config.name = config_dict['name']
    if 'description' in config_dict:
        config.description = config_dict['description']

    # 数据配置
    if 'data' in config_dict:
        for key, value in config_dict['data'].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

    # 模型配置
    if 'model' in config_dict:
        for key, value in config_dict['model'].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    # 损失配置
    if 'loss' in config_dict:
        for key, value in config_dict['loss'].items():
            if hasattr(config.loss, key):
                setattr(config.loss, key, value)

    # 训练配置
    if 'train' in config_dict:
        for key, value in config_dict['train'].items():
            if hasattr(config.train, key):
                setattr(config.train, key, value)

    # 消融实验配置
    if 'ablation' in config_dict:
        config.ablation.update(config_dict['ablation'])

    # 输出路径
    if 'output_dir' in config_dict:
        config.output_dir = config_dict['output_dir']
    if 'checkpoint_dir' in config_dict:
        config.checkpoint_dir = config_dict['checkpoint_dir']

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"  Experiment: {config.name}")
    logger.info(f"  Description: {config.description}")

    return config


def save_config(config: ExperimentConfig, save_path: str):
    """
    保存配置到YAML文件

    Args:
        config: ExperimentConfig对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为字典
    config_dict = {
        'name': config.name,
        'description': config.description,
        'data': {
            k: v for k, v in config.data.__dict__.items()
        },
        'model': {
            k: v for k, v in config.model.__dict__.items()
        },
        'loss': {
            k: v for k, v in config.loss.__dict__.items()
        },
        'train': {
            k: v for k, v in config.train.__dict__.items()
        },
        'ablation': config.ablation,
        'output_dir': config.output_dir,
        'checkpoint_dir': config.checkpoint_dir,
    }

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {save_path}")


def create_default_configs():
    """创建默认配置文件"""
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)

    # === Ours-Full ===
    config_full = ExperimentConfig(
        name='ours_full',
        description='Full model with all components'
    )
    save_config(config_full, configs_dir / 'ours_full.yaml')

    # === Ours w/o Contrast ===
    config_wo_contrast = ExperimentConfig(
        name='ours_wo_contrast',
        description='Without multi-view contrastive learning'
    )
    config_wo_contrast.ablation['use_contrast'] = False
    config_wo_contrast.loss.alpha_contrast = 0.0
    save_config(config_wo_contrast, configs_dir / 'ours_wo_contrast.yaml')

    # === Ours w/o Align ===
    config_wo_align = ExperimentConfig(
        name='ours_wo_align',
        description='Without Entity-Item alignment'
    )
    config_wo_align.ablation['use_align'] = False
    config_wo_align.loss.beta_align = 0.0
    save_config(config_wo_align, configs_dir / 'ours_wo_align.yaml')

    # === Ours w/o Mask ===
    config_wo_mask = ExperimentConfig(
        name='ours_wo_mask',
        description='Without learnable mask'
    )
    config_wo_mask.ablation['use_mask'] = False
    config_wo_mask.model.use_mask = False
    config_wo_mask.loss.gamma_mask = 0.0
    save_config(config_wo_mask, configs_dir / 'ours_wo_mask.yaml')

    # === Ours CF-only ===
    config_cf_only = ExperimentConfig(
        name='ours_cf_only',
        description='Only use CF view'
    )
    config_cf_only.ablation['use_kg_view'] = False
    save_config(config_cf_only, configs_dir / 'ours_cf_only.yaml')

    # === Ours KG-only ===
    config_kg_only = ExperimentConfig(
        name='ours_kg_only',
        description='Only use KG view'
    )
    config_kg_only.ablation['use_cf_view'] = False
    save_config(config_kg_only, configs_dir / 'ours_kg_only.yaml')

    logger.info(f"Created default configs in {configs_dir}/")


if __name__ == '__main__':
    # 创建默认配置
    logging.basicConfig(level=logging.INFO)
    create_default_configs()

    # 测试加载
    config = load_config('configs/ours_full.yaml')
    print(f"\n✓ Loaded config: {config.name}")
    print(f"  Embedding dim: {config.model.embedding_dim}")
    print(f"  Batch size: {config.train.batch_size}")
    print(f"  Use contrast: {config.ablation['use_contrast']}")
