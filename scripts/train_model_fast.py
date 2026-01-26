#!/usr/bin/env python3
"""
Accelerated Training Script with Mixed Precision and Optimizations

Key optimizations:
- Mixed precision training (torch.cuda.amp)
- torch.compile for faster model execution (PyTorch 2.0+)
- Gradient accumulation support
- Already has: pin_memory, persistent_workers, gradient clipping

Usage:
    python scripts/train_model_fast.py --config configs/ours_full.yaml
    python scripts/train_model_fast.py --config configs/ours_full.yaml --grad-accum-steps 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import random
import string

from src.data import KnowledgeGraphBuilder, compute_frequency_mask, split_data, create_dataloaders
from src.model import KnowledgeEnhancedRecModel, RecommendationLoss
from src.utils import load_config, evaluate_ranking_batched

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_run_id(model_name, dataset_name):
    """生成运行ID（类似RecBole格式）"""
    now = datetime.now()
    timestamp_str = now.strftime("%b-%d-%Y_%H-%M-%S")
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    short_hash = ''.join(random.choices(string.hexdigits.lower(), k=6))
    return timestamp_str, short_hash, date_str, time_str


class AcceleratedTrainer:
    """加速版训练器（混合精度 + 梯度累积 + 批量评估）"""

    def __init__(self, config, model, criterion, optimizer, device,
                 use_amp=True, grad_accum_steps=1, use_compile=True, eval_batch_size=512):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.eval_batch_size = eval_batch_size  # 评估批量大小（A800可以设更大）

        # 混合精度训练
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("✓ Mixed precision training enabled (AMP)")
        else:
            self.scaler = None
            logger.info("✗ Mixed precision disabled (CPU or disabled)")

        # torch.compile加速（PyTorch 2.0+）
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("✓ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"✗ torch.compile failed: {e}")
        else:
            logger.info("✗ torch.compile not available")

        # 生成运行ID
        model_name = f"{config.name}_fast"
        dataset_name = "ml-1m"
        timestamp_str, short_hash, date_str, time_str = generate_run_id(model_name, dataset_name)

        self.run_id = f"{model_name}-{dataset_name}-{timestamp_str}-{short_hash}"
        self.checkpoint_id = f"{model_name}_{dataset_name}_{date_str}_{time_str}"

        # 创建输出目录
        log_dir = Path("log") / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"{self.run_id}.log"

        tensorboard_dir = Path("log_tensorboard") / self.run_id
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)

        self.output_dir = Path(config.output_dir) / "ours" / self.checkpoint_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 设置文件日志
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("="*80)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Log file: {self.log_file}")
        logger.info(f"TensorBoard: {tensorboard_dir}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")
        logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
        logger.info(f"Effective batch size: {config.train.batch_size * grad_accum_steps}")
        logger.info("")
        logger.info("Training Configuration:")
        logger.info(f"  Random seed: {config.train.random_seed}")
        logger.info(f"  Epochs: {config.train.num_epochs}")
        logger.info(f"  Learning rate: {config.train.learning_rate}")
        logger.info(f"  Early stop patience: {config.train.early_stop_patience}")
        logger.info("")
        logger.info("Evaluation Configuration:")
        logger.info(f"  Eval mode: {config.train.eval_mode}")
        if config.train.eval_mode == 'uni100':
            logger.info(f"  Num negatives: {config.train.eval_num_neg} (1 pos + {config.train.eval_num_neg} neg)")
        logger.info(f"  Eval every: {config.train.eval_every} epoch(s)")
        logger.info(f"  Eval batch size: {eval_batch_size} (GPU并行评估)")
        logger.info(f"  Metrics: NDCG@[5,10,20], Recall@[5,10,20], Precision@[5,10,20], Hit@[5,10,20]")
        logger.info("="*80)

        # 最佳指标
        self.best_ndcg = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = defaultdict(list)

    def train_epoch(self, train_loader, hetero_graph, cf_edge_index, epoch):
        """训练一个epoch（带混合精度和梯度累积）"""
        self.model.train()

        total_loss = 0.0
        loss_components = defaultdict(float)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward（混合精度）
            with autocast(enabled=self.use_amp):
                outputs = self.model(hetero_graph, cf_edge_index)

                # 获取batch embedding
                user_ids = batch['user_id']
                pos_item_ids = batch['pos_item_id']
                neg_item_ids = batch['neg_item_ids']

                user_emb_fused = outputs['user_fused'][user_ids]
                pos_item_emb = outputs['item_fused'][pos_item_ids]
                neg_item_emb = outputs['item_fused'][neg_item_ids]

                # 计算损失
                loss_dict = self.criterion(
                    user_emb_fused=user_emb_fused,
                    pos_item_emb=pos_item_emb,
                    neg_item_emb=neg_item_emb,
                    user_emb_cf=outputs['user_cf'][user_ids] if outputs['user_cf'] is not None else None,
                    user_emb_kg=outputs['user_kg'][user_ids] if outputs['user_kg'] is not None else None,
                    entity_emb=outputs['entity_emb'],
                    item_emb_kg=outputs['item_kg'],
                    entity_item_edges=hetero_graph['entity', 'describes', 'item'].edge_index,
                    mask=outputs['mask']
                )

                loss = loss_dict['loss'] / self.grad_accum_steps  # 梯度累积：平均loss

            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf detected in loss! Loss components:")
                for key, value in loss_dict.items():
                    logger.error(f"  {key}: {value.item()}")
                raise ValueError("NaN/Inf detected in loss")

            # Backward（混合精度）
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积：每grad_accum_steps步更新一次
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.use_amp:
                    # Gradient clipping（混合精度下需要unscale）
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # 记录
            total_loss += loss.item() * self.grad_accum_steps  # 还原真实loss
            for key, value in loss_dict.items():
                loss_components[key] += value.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item() * self.grad_accum_steps:.4f}",
                'L_rec': f"{loss_dict['L_rec'].item():.4f}"
            })

        # 平均损失
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return avg_loss, avg_components

    @torch.no_grad()
    def evaluate(self, val_loader, hetero_graph, cf_edge_index, test_user_items, train_user_items, val_user_items=None):
        """评估模型（混合精度 + 批量并行评估）

        Args:
            val_user_items: 验证集用户物品（评估测试集时必须提供，用于排除）
        """
        self.model.eval()

        # Forward获取所有embeddings（混合精度）
        with autocast(enabled=self.use_amp):
            outputs = self.model(hetero_graph, cf_edge_index)
            user_emb = outputs['user_fused']
            item_emb = outputs['item_fused']

        # 计算指标（使用批量并行评估）
        eval_mode = self.config.train.eval_mode
        random_seed = self.config.train.random_seed

        # Only use eval_num_neg in uni100 mode
        if eval_mode == 'uni100':
            eval_num_neg = self.config.train.eval_num_neg
        else:
            eval_num_neg = 99  # Default, not used in full mode

        metrics = evaluate_ranking_batched(
            user_emb=user_emb,
            item_emb=item_emb,
            test_user_items=test_user_items,
            train_user_items=train_user_items,
            val_user_items=val_user_items,  # 添加val排除
            k_list=[5, 10, 20],
            exclude_train=True,
            mode=eval_mode,
            num_neg=eval_num_neg,
            seed=random_seed,
            batch_size=self.eval_batch_size  # 使用批量评估
        )

        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        hetero_graph,
        cf_edge_index,
        val_user_items,
        test_user_items,
        train_user_items
    ):
        """完整训练流程"""
        logger.info(f"Starting accelerated training for {self.config.name}")
        logger.info(f"  Num epochs: {self.config.train.num_epochs}")
        logger.info(f"  Batch size: {self.config.train.batch_size}")
        logger.info(f"  Gradient accumulation: {self.grad_accum_steps}")
        logger.info(f"  Effective batch size: {self.config.train.batch_size * self.grad_accum_steps}")
        logger.info(f"  Learning rate: {self.config.train.learning_rate}")

        for epoch in range(1, self.config.train.num_epochs + 1):
            # 训练
            avg_loss, loss_components = self.train_epoch(
                train_loader, hetero_graph, cf_edge_index, epoch
            )

            # 记录训练loss
            self.history['train_loss'].append(avg_loss)
            for key, value in loss_components.items():
                self.history[f'train_{key}'].append(value)

            # Log
            if epoch % self.config.train.log_every == 0:
                logger.info(f"Epoch {epoch}/{self.config.train.num_epochs}")
                logger.info(f"  Train Loss: {avg_loss:.4f}")
                logger.info(f"    L_rec: {loss_components['L_rec']:.4f}")
                if loss_components['L_contrast'] > 0:
                    logger.info(f"    L_contrast: {loss_components['L_contrast']:.4f}")
                if loss_components['L_align'] > 0:
                    logger.info(f"    L_align: {loss_components['L_align']:.4f}")
                if loss_components['L_mask'] > 0:
                    logger.info(f"    L_mask: {loss_components['L_mask']:.4f}")

                # TensorBoard
                self.writer.add_scalar('Loss/train', avg_loss, epoch)
                for key, value in loss_components.items():
                    self.writer.add_scalar(f'Loss/{key}', value, epoch)

            # 评估
            if epoch % self.config.train.eval_every == 0:
                val_metrics = self.evaluate(
                    val_loader, hetero_graph, cf_edge_index,
                    val_user_items, train_user_items
                )

                # 记录
                for key, value in val_metrics.items():
                    self.history[f'val_{key}'].append(value)
                    self.writer.add_scalar(f'Val/{key}', value, epoch)

                logger.info(f"  Val Metrics:")
                logger.info(f"    NDCG@10: {val_metrics['NDCG@10']:.4f}")
                logger.info(f"    Recall@10: {val_metrics['Recall@10']:.4f}")

                # Early stopping
                if val_metrics['NDCG@10'] > self.best_ndcg:
                    self.best_ndcg = val_metrics['NDCG@10']
                    self.best_epoch = epoch
                    self.patience_counter = 0

                    # 保存最佳模型
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"  ✓ New best model! NDCG@10={self.best_ndcg:.4f}")
                else:
                    self.patience_counter += 1

                    if self.patience_counter >= self.config.train.early_stop_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # 定期保存
            if epoch % self.config.train.save_every == 0:
                self.save_checkpoint(epoch, {}, is_best=False)

        # 最终测试（必须排除train和val）
        logger.info("\nFinal Test Evaluation:")
        test_metrics = self.evaluate(
            test_loader, hetero_graph, cf_edge_index,
            test_user_items, train_user_items, val_user_items  # 添加val排除
        )

        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # 保存历史记录
        self.save_history(test_metrics)

        return test_metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_ndcg': self.best_ndcg,
            'metrics': metrics,
            'config': self.config.__dict__
        }

        if is_best:
            timestamp_str = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
            filename = f"{self.config.name}-{timestamp_str}.pth"
            path = self.checkpoint_dir / filename
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)
        logger.info(f"  Saved checkpoint: {path}")

    def save_history(self, final_test_metrics):
        """保存训练历史"""
        history_path = self.output_dir / 'history.json'

        history_dict = dict(self.history)
        history_dict['final_test_metrics'] = final_test_metrics
        history_dict['best_epoch'] = self.best_epoch
        history_dict['best_ndcg'] = self.best_ndcg

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Saved training history: {history_path}")


def prepare_user_items_dict(df, user_id_map, item_id_map):
    """准备user_items字典（用于评估）"""
    user_items = defaultdict(list)

    for _, row in df.iterrows():
        user_id = user_id_map[row['user_id:token']]
        item_id = item_id_map[row['item_id:token']]
        user_items[user_id].append(item_id)

    return dict(user_items)


def main(args):
    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    torch.manual_seed(config.train.random_seed)
    np.random.seed(config.train.random_seed)

    # 设置设备
    device = torch.device(config.train.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # === 1. 分割数据（必须先分割，再构建图！）===
    logger.info("\n=== Splitting Data ===")
    train_df, val_df, test_df = split_data(
        config.data.inter_path,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        time_based=config.data.time_based_split,
        random_seed=config.train.random_seed,
        per_user_split=config.data.per_user_split
    )

    # === 2. 构建图（只用训练集，避免数据泄露！）===
    logger.info("\n=== Building Knowledge Graph ===")
    graph_builder = KnowledgeGraphBuilder(
        item_kg_path=config.data.item_kg_path,
        user_kg_path=config.data.user_kg_path,
        inter_path=config.data.inter_path,
        min_rating=config.data.min_rating
    )

    # CRITICAL: 只用训练集构建图（避免test/val数据泄露）
    hetero_graph, cf_edge_index, stats = graph_builder.build_hetero_graph(train_inter_df=train_df)

    # 移动图到设备
    hetero_graph = hetero_graph.to(device)
    cf_edge_index = cf_edge_index.to(device)

    # === 3. 计算Mask初始值 ===
    if config.model.use_mask:
        mask_init = compute_frequency_mask(
            stats['entity_frequency'],
            stats['entity_id_map'],
            min_freq=config.model.mask_min_freq,
            max_freq=config.model.mask_max_freq
        ).to(device)
    else:
        mask_init = None

    # === 4. 创建DataLoaders ===
    logger.info("\n=== Creating DataLoaders ===")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        stats['user_id_map'],
        stats['item_id_map'],
        batch_size=config.train.batch_size,
        num_negatives=config.train.num_negatives,
        num_workers=config.train.num_workers
    )

    # 准备评估用的user_items字典
    train_user_items = prepare_user_items_dict(train_df, stats['user_id_map'], stats['item_id_map'])
    val_user_items = prepare_user_items_dict(val_df, stats['user_id_map'], stats['item_id_map'])
    test_user_items = prepare_user_items_dict(test_df, stats['user_id_map'], stats['item_id_map'])

    # === 5. 创建模型 ===
    logger.info("\n=== Creating Model ===")
    model = KnowledgeEnhancedRecModel(
        num_users=stats['num_users'],
        num_items=stats['num_items'],
        num_entities=stats['num_entities'],
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_gnn_layers,
        gat_heads=config.model.gat_heads,
        dropout=config.model.dropout,
        use_mask=config.model.use_mask,
        mask_init=mask_init,
        use_cf_view=config.ablation['use_cf_view'],
        use_kg_view=config.ablation['use_kg_view']
    ).to(device)

    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # === 6. 损失函数 ===
    criterion = RecommendationLoss(
        alpha_contrast=config.loss.alpha_contrast,
        beta_align=config.loss.beta_align,
        gamma_mask=config.loss.gamma_mask,
        temperature_rec=config.loss.temperature_rec,
        temperature_contrast=config.loss.temperature_contrast,
        lambda_sparse=config.loss.lambda_sparse,
        lambda_entropy=config.loss.lambda_entropy,
        num_neg_align=config.loss.num_neg_align,
        use_contrast=config.ablation['use_contrast'],
        use_align=config.ablation['use_align'],
        use_mask=config.ablation['use_mask']
    )

    # === 7. 优化器 ===
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )

    # === 8. 训练（加速版）===
    logger.info("\n=== Starting Accelerated Training ===")
    trainer = AcceleratedTrainer(
        config, model, criterion, optimizer, device,
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps,
        use_compile=args.use_compile,
        eval_batch_size=args.eval_batch_size
    )

    test_metrics = trainer.train(
        train_loader, val_loader, test_loader,
        hetero_graph, cf_edge_index,
        val_user_items, test_user_items, train_user_items
    )

    logger.info("\n=== Training Complete ===")
    logger.info(f"Best epoch: {trainer.best_epoch}")
    logger.info(f"Best Val NDCG@10: {trainer.best_ndcg:.4f}")
    logger.info("\nFinal Test Metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Knowledge-Enhanced Recommendation Model (Accelerated)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/ours_full.yaml)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use mixed precision training (default: True)')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                        help='Disable mixed precision training')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--use-compile', action='store_true', default=True,
                        help='Use torch.compile (default: True)')
    parser.add_argument('--no-compile', dest='use_compile', action='store_false',
                        help='Disable torch.compile')
    parser.add_argument('--eval-batch-size', type=int, default=512,
                        help='Batch size for evaluation (default: 512, A800 can handle 1024+)')

    args = parser.parse_args()

    main(args)
