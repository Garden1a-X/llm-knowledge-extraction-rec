#!/usr/bin/env python3
"""
Loss functions for recommendation

Implements InfoNCE, multi-view contrastive, and entity alignment loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def info_nce_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
    temperature: float = 0.2
) -> torch.Tensor:
    """
    InfoNCE contrastive learning loss (recommendation task).

    Args:
        user_emb: [batch_size, dim] - user embedding
        pos_item_emb: [batch_size, dim] - positive item embedding
        neg_item_emb: [batch_size, num_neg, dim] - negative item embeddings
        temperature: temperature coefficient

    Returns:
        loss: scalar
    """
    # Positive pair score
    pos_score = (user_emb * pos_item_emb).sum(dim=-1) / temperature  # [batch_size]

    # Negative pairs scores
    neg_score = torch.bmm(
        neg_item_emb,
        user_emb.unsqueeze(-1)
    ).squeeze(-1) / temperature  # [batch_size, num_neg]

    # InfoNCE: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
    # Equivalent to CrossEntropy where positive sample label = 0
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)  # [batch_size, 1+num_neg]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)

    return loss


def bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor
) -> torch.Tensor:
    """
    BPR (Bayesian Personalized Ranking) loss.

    Args:
        pos_score: [batch_size] or [batch_size, 1] - positive sample scores
        neg_score: [batch_size, num_neg] - negative sample scores

    Returns:
        loss: scalar
    """
    if pos_score.dim() == 1:
        pos_score = pos_score.unsqueeze(1)  # [batch_size, 1]

    # BPR loss: -log(sigmoid(pos - neg))
    loss = -F.logsigmoid(pos_score - neg_score).mean()

    return loss


def multiview_contrastive_loss(
    emb_view1: torch.Tensor,
    emb_view2: torch.Tensor,
    temperature: float = 0.1,
    batch_wise: bool = True
) -> torch.Tensor:
    """
    Multi-view contrastive learning loss.

    Goal: encourage the two views to learn similar yet complementary user representations.

    Args:
        emb_view1: [batch_size or num_users, dim] - view 1 embedding (CF)
        emb_view2: [batch_size or num_users, dim] - view 2 embedding (KG)
        temperature: temperature coefficient
        batch_wise: whether to contrast only within the batch (recommended, faster)

    Returns:
        loss: scalar
    """
    # L2 normalization
    emb_view1 = F.normalize(emb_view1, dim=-1)
    emb_view2 = F.normalize(emb_view2, dim=-1)

    batch_size = emb_view1.size(0)

    # Positive pairs: two views of the same user
    pos_sim = (emb_view1 * emb_view2).sum(dim=-1) / temperature  # [batch_size]

    if batch_wise:
        # 负样本：batch内其他user的cross-view相似度
        neg_sim = emb_view1 @ emb_view2.T / temperature  # [batch_size, batch_size]

        # InfoNCE formulation
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1+batch_size]
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
    else:
        # 全局对比（更准确但更慢）
        # 正样本分数
        pos_logits = pos_sim

        # 负样本：所有其他user
        neg_sim_all = emb_view1 @ emb_view2.T / temperature  # [batch_size, batch_size]

        # 构建logits矩阵
        logits = neg_sim_all

        # 对角线是正样本
        labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

    return loss


def entity_item_alignment_loss(
    entity_emb: torch.Tensor,
    item_emb: torch.Tensor,
    edge_index: torch.Tensor,
    num_neg: int = 5
) -> torch.Tensor:
    """
    Entity-Item对齐损失

    让Entity和它描述的Item在embedding空间中接近

    Args:
        entity_emb: [num_entities, dim] - Entity embeddings
        item_emb: [num_items, dim] - Item embeddings
        edge_index: [2, num_edges] - (entity_describes_item边)
            edge_index[0]: entity IDs
            edge_index[1]: item IDs
        num_neg: 负采样数量

    Returns:
        loss: scalar
    """
    # Handle empty edge_index (e.g., when Item KG is empty in ablation)
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=entity_emb.device, requires_grad=True)

    entity_ids = edge_index[0]  # [num_edges]
    item_ids = edge_index[1]    # [num_edges]

    num_edges = entity_ids.size(0)
    num_items = item_emb.size(0)

    # Positive pairs
    pos_entity = entity_emb[entity_ids]  # [num_edges, dim]
    pos_item = item_emb[item_ids]        # [num_edges, dim]
    pos_score = (pos_entity * pos_item).sum(dim=-1)  # [num_edges]

    # Negative sampling（随机采样）
    neg_items = torch.randint(
        0, num_items,
        (num_edges, num_neg),
        device=item_emb.device
    )
    neg_item_emb = item_emb[neg_items]  # [num_edges, num_neg, dim]

    neg_score = torch.bmm(
        neg_item_emb,
        pos_entity.unsqueeze(-1)
    ).squeeze(-1)  # [num_edges, num_neg]

    # BPR loss
    loss = bpr_loss(pos_score, neg_score)

    return loss


def mask_regularization(
    mask: torch.Tensor,
    lambda_sparse: float = 1.0,
    lambda_entropy: float = 0.1
) -> torch.Tensor:
    """
    Mask正则化损失

    目标：
    1. 稀疏性：大部分Entity保留（mask≈1）
    2. 确定性：避免模棱两可（mask接近0或1）

    Args:
        mask: [num_entities] - sigmoid输出，范围[0, 1]
        lambda_sparse: 稀疏正则权重
        lambda_entropy: 熵正则权重

    Returns:
        loss: scalar
    """
    # L1稀疏正则：鼓励大部分=1（不mask）
    L_sparse = (1 - mask).sum()

    # 熵正则：鼓励接近0或1（最小化熵）
    eps = 1e-8
    entropy = -(
        mask * torch.log(mask + eps) +
        (1 - mask) * torch.log(1 - mask + eps)
    ).mean()

    # 组合损失
    loss = lambda_sparse * L_sparse - lambda_entropy * entropy

    return loss


class RecommendationLoss(nn.Module):
    """完整的推荐损失（组合所有损失）"""

    def __init__(
        self,
        alpha_contrast: float = 0.1,
        beta_align: float = 0.05,
        gamma_mask: float = 0.01,
        temperature_rec: float = 0.2,
        temperature_contrast: float = 0.1,
        lambda_sparse: float = 1.0,
        lambda_entropy: float = 0.1,
        num_neg_align: int = 5,
        use_contrast: bool = True,
        use_align: bool = True,
        use_mask: bool = True,
    ):
        """
        Args:
            alpha_contrast: 多视图对比损失权重
            beta_align: Entity-Item对齐损失权重
            gamma_mask: Mask正则化权重
            temperature_rec: InfoNCE推荐损失温度
            temperature_contrast: 多视图对比损失温度
            lambda_sparse: Mask稀疏正则权重
            lambda_entropy: Mask熵正则权重
            num_neg_align: 对齐损失负采样数
            use_contrast: 是否使用多视图对比
            use_align: 是否使用Entity-Item对齐
            use_mask: 是否使用Mask正则
        """
        super().__init__()

        self.alpha = alpha_contrast
        self.beta = beta_align
        self.gamma = gamma_mask

        self.temp_rec = temperature_rec
        self.temp_contrast = temperature_contrast

        self.lambda_sparse = lambda_sparse
        self.lambda_entropy = lambda_entropy

        self.num_neg_align = num_neg_align

        self.use_contrast = use_contrast
        self.use_align = use_align
        self.use_mask = use_mask

    def forward(
        self,
        user_emb_fused: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        user_emb_cf: Optional[torch.Tensor] = None,
        user_emb_kg: Optional[torch.Tensor] = None,
        entity_emb: Optional[torch.Tensor] = None,
        item_emb_kg: Optional[torch.Tensor] = None,
        entity_item_edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算完整损失

        Args:
            user_emb_fused: [batch_size, dim] - 融合后的用户embedding
            pos_item_emb: [batch_size, dim] - 正样本item
            neg_item_emb: [batch_size, num_neg, dim] - 负样本items
            user_emb_cf: [num_users or batch_size, dim] - CF视图用户embedding
            user_emb_kg: [num_users or batch_size, dim] - KG视图用户embedding
            entity_emb: [num_entities, dim] - Entity embeddings
            item_emb_kg: [num_items, dim] - KG视图Item embeddings
            entity_item_edges: [2, num_edges] - Entity-Item边
            mask: [num_entities] - 学习的mask

        Returns:
            loss_dict: {
                'loss': total_loss,
                'L_rec': ...,
                'L_contrast': ...,
                'L_align': ...,
                'L_mask': ...
            }
        """
        loss_dict = {}

        # === 1. 主损失：InfoNCE推荐 ===
        L_rec = info_nce_loss(
            user_emb_fused,
            pos_item_emb,
            neg_item_emb,
            temperature=self.temp_rec
        )
        loss_dict['L_rec'] = L_rec

        # === 2. 多视图对比损失 ===
        if self.use_contrast and user_emb_cf is not None and user_emb_kg is not None:
            L_contrast = multiview_contrastive_loss(
                user_emb_cf,
                user_emb_kg,
                temperature=self.temp_contrast,
                batch_wise=True  # batch内对比（更快）
            )
            loss_dict['L_contrast'] = L_contrast
        else:
            L_contrast = torch.tensor(0.0, device=L_rec.device)
            loss_dict['L_contrast'] = L_contrast

        # === 3. Entity-Item对齐损失 ===
        if self.use_align and entity_emb is not None and item_emb_kg is not None and entity_item_edges is not None:
            L_align = entity_item_alignment_loss(
                entity_emb,
                item_emb_kg,
                entity_item_edges,
                num_neg=self.num_neg_align
            )
            loss_dict['L_align'] = L_align
        else:
            L_align = torch.tensor(0.0, device=L_rec.device)
            loss_dict['L_align'] = L_align

        # === 4. Mask正则化 ===
        if self.use_mask and mask is not None:
            L_mask = mask_regularization(
                mask,
                lambda_sparse=self.lambda_sparse,
                lambda_entropy=self.lambda_entropy
            )
            loss_dict['L_mask'] = L_mask
        else:
            L_mask = torch.tensor(0.0, device=L_rec.device)
            loss_dict['L_mask'] = L_mask

        # === 总损失 ===
        total_loss = (
            L_rec
            + self.alpha * L_contrast
            + self.beta * L_align
            + self.gamma * L_mask
        )
        loss_dict['loss'] = total_loss

        return loss_dict


if __name__ == '__main__':
    # 测试损失函数
    torch.manual_seed(42)

    batch_size = 64
    dim = 128
    num_neg = 5

    # 测试数据
    user_emb = torch.randn(batch_size, dim)
    pos_item_emb = torch.randn(batch_size, dim)
    neg_item_emb = torch.randn(batch_size, num_neg, dim)

    # 1. InfoNCE
    loss_rec = info_nce_loss(user_emb, pos_item_emb, neg_item_emb)
    print(f"InfoNCE loss: {loss_rec.item():.4f}")

    # 2. 多视图对比
    user_cf = torch.randn(batch_size, dim)
    user_kg = torch.randn(batch_size, dim)
    loss_contrast = multiview_contrastive_loss(user_cf, user_kg)
    print(f"Contrastive loss: {loss_contrast.item():.4f}")

    # 3. Mask正则
    mask = torch.sigmoid(torch.randn(400))
    loss_mask = mask_regularization(mask)
    print(f"Mask regularization: {loss_mask.item():.4f}")

    # 4. 完整损失
    criterion = RecommendationLoss()
    loss_dict = criterion(
        user_emb_fused=user_emb,
        pos_item_emb=pos_item_emb,
        neg_item_emb=neg_item_emb,
        user_emb_cf=user_cf,
        user_emb_kg=user_kg,
        mask=mask
    )

    print(f"\n✓ Total loss: {loss_dict['loss'].item():.4f}")
    for key, value in loss_dict.items():
        if key != 'loss':
            print(f"  {key}: {value.item():.4f}")
