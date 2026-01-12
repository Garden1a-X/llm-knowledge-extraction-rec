#!/usr/bin/env python3
"""
Training script for Knowledge-Enhanced Recommendation Model

Usage:
    python scripts/train_model.py --config configs/ours_full.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from src.data import KnowledgeGraphBuilder, compute_frequency_mask, split_data, create_dataloaders
from src.model import KnowledgeEnhancedRecModel, RecommendationLoss
from src.utils import load_config, evaluate_ranking

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器"""

    def __init__(self, config, model, criterion, optimizer, device):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # 创建输出目录
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(config.checkpoint_dir) / config.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')

        # 最佳指标
        self.best_ndcg = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # 历史记录
        self.history = defaultdict(list)

    def train_epoch(self, train_loader, hetero_graph, cf_edge_index, epoch):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        loss_components = defaultdict(float)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward
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

            loss = loss_dict['loss']

            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf detected in loss! Loss components:")
                for key, value in loss_dict.items():
                    logger.error(f"  {key}: {value.item()}")
                raise ValueError("NaN/Inf detected in loss")

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            for key, value in loss_dict.items():
                loss_components[key] += value.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'L_rec': f"{loss_dict['L_rec'].item():.4f}"
            })

        # 平均损失
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return avg_loss, avg_components

    @torch.no_grad()
    def evaluate(self, val_loader, hetero_graph, cf_edge_index, test_user_items, train_user_items):
        """评估模型"""
        self.model.eval()

        # Forward获取所有embeddings
        outputs = self.model(hetero_graph, cf_edge_index)
        user_emb = outputs['user_fused']
        item_emb = outputs['item_fused']

        # 计算指标
        metrics = evaluate_ranking(
            user_emb=user_emb,
            item_emb=item_emb,
            test_user_items=test_user_items,
            train_user_items=train_user_items,
            k_list=[5, 10, 20],
            exclude_train=True
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
        logger.info(f"Starting training for {self.config.name}")
        logger.info(f"  Num epochs: {self.config.train.num_epochs}")
        logger.info(f"  Batch size: {self.config.train.batch_size}")
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

        # 最终测试
        logger.info("\nFinal Test Evaluation:")
        test_metrics = self.evaluate(
            test_loader, hetero_graph, cf_edge_index,
            test_user_items, train_user_items
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
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

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

    # === 1. 构建图 ===
    logger.info("\n=== Building Knowledge Graph ===")
    graph_builder = KnowledgeGraphBuilder(
        item_kg_path=config.data.item_kg_path,
        user_kg_path=config.data.user_kg_path,
        inter_path=config.data.inter_path,
        min_rating=config.data.min_rating
    )

    hetero_graph, cf_edge_index, stats = graph_builder.build_hetero_graph()

    # 移动图到设备
    hetero_graph = hetero_graph.to(device)
    cf_edge_index = cf_edge_index.to(device)

    # === 2. 计算Mask初始值 ===
    if config.model.use_mask:
        mask_init = compute_frequency_mask(
            stats['entity_frequency'],
            stats['entity_id_map'],
            min_freq=config.model.mask_min_freq,
            max_freq=config.model.mask_max_freq
        ).to(device)
    else:
        mask_init = None

    # === 3. 分割数据 ===
    logger.info("\n=== Splitting Data ===")
    train_df, val_df, test_df = split_data(
        config.data.inter_path,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        time_based=config.data.time_based_split,
        random_seed=config.train.random_seed
    )

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

    # === 8. 训练 ===
    logger.info("\n=== Starting Training ===")
    trainer = Trainer(config, model, criterion, optimizer, device)

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
    parser = argparse.ArgumentParser(description='Train Knowledge-Enhanced Recommendation Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/ours_full.yaml)')

    args = parser.parse_args()

    main(args)
