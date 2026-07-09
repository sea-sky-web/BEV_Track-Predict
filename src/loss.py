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
    
    处理流程：
    1. 将目标热图自适应池化到预测尺寸
    2. 对目标应用高斯卷积平滑
    3. 计算预测与平滑目标的 MSE
    
    这种方法优于直接 MSE，因为：
    - 高斯模糊提供软标签而非硬标签
    - 允许预测在目标周围有梯度
    - 更容易优化
    
    Attributes:
        无（无需初始化参数）
        
    References:
        基于 MVDet: A Baseline for Multi-View 3D Pedestrian Detection
        (https://github.com/hou-yz/MVDet)
    """
    
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        计算高斯 MSE 损失
        
        Args:
            pred: 预测热图 (B, C, H, W)
                  通常已通过 sigmoid 归一化到 [0, 1]
                  
            target: 目标热图 (B, C, Ht, Wt)
                   通常为二值图像（0 或 1）
                   Ht, Wt 可能与 H, W 不同
                   
            kernel: 高斯核 (1, 1, K, K)
                   由 build_gaussian_kernel_2d() 生成
                   
        Returns:
            torch.Tensor: 标量损失值
            
        Shape:
            - pred: (B, C, H, W)
            - target: (B, C, Ht, Wt) 其中 Ht, Wt >= H, W
            - kernel: (1, 1, K, K)
            - output: 标量
            
        Example:
            >>> import torch
            >>> from utils import build_gaussian_kernel_2d
            >>> criterion = GaussianMSE()
            >>> pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
            >>> target = torch.randint(0, 2, (2, 1, 128, 128)).float()
            >>> kernel = build_gaussian_kernel_2d(11, 2.5, torch.device("cpu"))
            >>> loss = criterion(pred, target, kernel)
            >>> print(f"Loss: {loss.item():.6f}")
        """
        B, C, H, W = pred.shape
        
        # 步骤 1: 自适应池化目标到预测尺寸
        # 使用 max 池化保留目标信号，避免平均化稀疏标签
        target = F.adaptive_max_pool2d(target, output_size=(H, W))
        
        # 步骤 2: 应用高斯卷积平滑
        # 将多通道重塑为 (B*C, 1, H, W) 以支持单通道卷积
        tgt = target.reshape(B * C, 1, H, W)
        
        # 确保核的数据类型和设备与目标一致
        k = kernel.to(dtype=tgt.dtype, device=tgt.device)
        
        # 计算填充大小
        pad = (k.shape[-1] - 1) // 2
        
        # 应用高斯卷积
        tgt = F.conv2d(tgt, k, padding=pad)
        
        # 恢复为多通道形状
        tgt = tgt.reshape(B, C, H, W)
        
        # 步骤 3: 计算 MSE 损失
        return F.mse_loss(pred, tgt)


class WeightedGaussianMSE(nn.Module):
    """
    带权重的高斯 MSE 损失（可选的高级版本）
    
    允许对正样本和负样本给予不同权重，
    适合处理不平衡的检测任务。
    
    Args:
        pos_weight: 正样本权重，默认 1.0
        neg_weight: 负样本权重，默认 1.0
    """
    
    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.base_criterion = GaussianMSE()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        计算带权重的高斯 MSE 损失
        
        Args:
            pred: 预测热图 (B, C, H, W)
            target: 目标热图 (B, C, Ht, Wt)
            kernel: 高斯核 (1, 1, K, K)
            
        Returns:
            torch.Tensor: 标量损失值
        """
        B, C, H, W = pred.shape
        target = F.adaptive_max_pool2d(target, output_size=(H, W))
        
        tgt = target.reshape(B * C, 1, H, W)
        k = kernel.to(dtype=tgt.dtype, device=tgt.device)
        pad = (k.shape[-1] - 1) // 2
        tgt = F.conv2d(tgt, k, padding=pad)
        tgt = tgt.reshape(B, C, H, W)
        
        # 计算加权 MSE
        diff = (pred - tgt) ** 2
        
        # 对正负样本应用不同权重
        pos_mask = tgt > 0.5
        neg_mask = ~pos_mask
        
        weighted_diff = diff.clone()
        weighted_diff[pos_mask] *= self.pos_weight
        weighted_diff[neg_mask] *= self.neg_weight
        
        return weighted_diff.mean()


def create_loss_criterion(
    weighted: bool = False,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    loss_type: str = "mse",
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
) -> nn.Module:
    if loss_type == "focal":
        return PenaltyReducedFocalLoss(alpha=focal_alpha, beta=focal_beta)
    if weighted:
        return WeightedGaussianMSE(pos_weight=pos_weight, neg_weight=neg_weight)
    return GaussianMSE()


class PenaltyReducedFocalLoss(nn.Module):
    """CenterNet-style penalty-reduced focal loss for heatmap regression.

    Reference: Law & Deng, "CornerNet" (arXiv 1808.01244), Section 3.
    Used by all modern multi-view detectors (QMVDet, EarlyBird, PVH, MSMVD).

    For positive locations (smoothed GT == 1):
        L = -(1 - p)^alpha * log(p)
    For negative locations (smoothed GT < 1):
        L = -(1 - Y)^beta * p^alpha * log(1 - p)
    where p = sigmoid(pred_logits), Y = smoothed GT.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = pred.shape

        target = F.adaptive_max_pool2d(target, output_size=(H, W))
        tgt = target.reshape(B * C, 1, H, W)
        k = kernel.to(dtype=tgt.dtype, device=tgt.device)
        pad = (k.shape[-1] - 1) // 2
        tgt = F.conv2d(tgt, k, padding=pad)
        tgt = tgt.reshape(B, C, H, W)
        # Clamp to [0,1]: overlapping Gaussians from nearby pedestrians can sum > 1.0
        tgt = tgt.clamp(max=1.0)

        p = torch.sigmoid(pred)
        p = p.clamp(1e-6, 1.0 - 1e-6)

        # ge(1-eps) instead of eq(1.0): conv2d floating point rarely produces exact 1.0
        pos_mask = tgt.ge(1.0 - 1e-4)
        neg_mask = ~pos_mask

        pos_loss = -((1.0 - p) ** self.alpha) * torch.log(p)
        neg_loss = -((1.0 - tgt) ** self.beta) * (p ** self.alpha) * torch.log(1.0 - p)

        loss = torch.where(pos_mask, pos_loss, neg_loss)
        num_pos = pos_mask.float().sum().clamp_min(1.0)
        return loss.sum() / num_pos
