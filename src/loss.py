# scripts/loss.py
"""
损失函数模块：对标 MVDet 的 GaussianMSE 损失
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMSE(nn.Module):
    """
    MVDet 风格的高斯 MSE 损失函数
    """
    
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        target = F.adaptive_max_pool2d(target, output_size=(H, W))
        tgt = target.reshape(B * C, 1, H, W)
        k = kernel.to(dtype=tgt.dtype, device=tgt.device)
        pad = (k.shape[-1] - 1) // 2
        tgt = F.conv2d(tgt, k, padding=pad)
        tgt = tgt.reshape(B, C, H, W)
        return F.mse_loss(pred, tgt)


class WeightedGaussianMSE(nn.Module):
    """
    带权重的高斯 MSE 损失
    """
    
    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        target = F.adaptive_max_pool2d(target, output_size=(H, W))
        tgt = target.reshape(B * C, 1, H, W)
        k = kernel.to(dtype=tgt.dtype, device=tgt.device)
        pad = (k.shape[-1] - 1) // 2
        tgt = F.conv2d(tgt, k, padding=pad)
        tgt = tgt.reshape(B, C, H, W)
        
        diff = (pred - tgt) ** 2
        pos_mask = tgt > 0.1
        neg_mask = ~pos_mask
        
        weighted_diff = diff.clone()
        weighted_diff[pos_mask] *= self.pos_weight
        weighted_diff[neg_mask] *= self.neg_weight
        
        return weighted_diff.mean()


class FocalLoss(nn.Module):
    """
    针对热图优化的 Focal Loss (Modified for heatmaps as in CornerNet/CenterNet)
    
    L = -1/N * sum(
        (1-p)^gamma * log(p)           if y=1
        (1-y)^beta * p^gamma * log(1-p) if y<1
    )
    
    Args:
        alpha: 正负样本权重 (CornerNet/CenterNet typically don't use alpha here)
        gamma: 难易样本权重，通常取 2
        beta: 负样本惩罚权重，通常取 4
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha  # alpha is gamma in standard Focal Loss terminology for heatmaps
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测热图 (B, C, H, W), 已过 sigmoid
            target: 目标热图 (B, C, Ht, Wt), 二值
            kernel: 高斯核 (1, 1, K, K)
        """
        B, C, H, W = pred.shape
        target = F.adaptive_max_pool2d(target, output_size=(H, W))
        tgt = target.reshape(B * C, 1, H, W)
        k = kernel.to(dtype=tgt.dtype, device=tgt.device)
        pad = (k.shape[-1] - 1) // 2
        tgt = F.conv2d(tgt, k, padding=pad)
        tgt = tgt.reshape(B, C, H, W)
        
        # 裁剪 pred 避免 log(0)
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        
        pos_inds = tgt.eq(1).float()
        neg_inds = tgt.lt(1).float()

        neg_weights = torch.pow(1 - tgt, self.beta)
        
        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def create_loss_criterion(
    loss_type: str = "mse",
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
) -> nn.Module:
    """
    工厂函数：创建损失函数
    
    Args:
        loss_type: 'mse', 'weighted_mse', 'focal'
        pos_weight: 正样本权重 (weighted_mse)
        neg_weight: 负样本权重 (weighted_mse)
        focal_alpha: Focal Loss alpha 参数 (gamma)
        focal_beta: Focal Loss beta 参数
    """
    if loss_type == "mse":
        return GaussianMSE()
    elif loss_type == "weighted_mse":
        return WeightedGaussianMSE(pos_weight=pos_weight, neg_weight=neg_weight)
    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, beta=focal_beta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
