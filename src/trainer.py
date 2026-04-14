# scripts/trainer.py
"""
训练器模块：训练循环、验证、日志记录和检查点管理
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import GaussianMSE
from utils import save_heat_png, build_gaussian_kernel_2d


class MVDetTrainer:
    """
    MVDet 风格的训练器
    
    负责完整的训练流程：
    - 前向/反向传播
    - 损失计算（BEV + 图像）
    - 日志记录
    - 模型检查点保存
    - 可视化
    
    Attributes:
        model: 神经网络模型
        device: 计算设备
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        scaler: AMP 梯度缩放器
        output_dir: 输出目录
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        output_dir: Path,
        amp_enabled: bool = False,
        freeze_bn: bool = False,
    ):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            output_dir: 输出目录（保存检查点、日志等）
            amp_enabled: 是否启用自动混合精度
            freeze_bn: 是否冻结 BatchNorm 层
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 损失函数
        self.criterion = GaussianMSE()
        
        # AMP 梯度缩放器
        self.scaler = torch.amp.GradScaler(
            device=str(device),
            enabled=(amp_enabled and device.type == "cuda")
        )
        self.amp_enabled = amp_enabled and device.type == "cuda"
        
        # 冻结 BatchNorm
        if freeze_bn:
            self._freeze_bn()
        
        # 全局步数计数
        self.global_step = 0
    
    def _freeze_bn(self):
        """冻结所有 BatchNorm 层"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        map_kernel: torch.Tensor,
        img_kernel: torch.Tensor,
        alpha: float = 1.0,
        log_every: int = 20,
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            map_kernel: BEV 热图的高斯核
            img_kernel: 图像热图的高斯核
            alpha: 图像损失的权重
            log_every: 每多少步打印一次日志
            
        Returns:
            dict: epoch 统计信息
                - "loss": 平均总损失
                - "bev_loss": 平均 BEV 损失
                - "img_loss": 平均图像损失
                - "pos_mse": 正样本 MSE
                - "aux_pos_mse": 辅助正样本 MSE
        """
        self.model.train()
        
        losses = []
        bev_losses = []
        img_losses = []
        pos_mses = []
        aux_pos_mses = []
        
        for batch_idx, (stems, x_views, map_gt, imgs_gt) in enumerate(train_loader):
            x_views = x_views.to(self.device, non_blocking=True)
            map_gt = map_gt.to(self.device, non_blocking=True)
            imgs_gt = imgs_gt.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                map_logits, imgs_logits = self.model(x_views)
                map_res = torch.sigmoid(map_logits)
                imgs_res = torch.sigmoid(imgs_logits)
                
                # BEV 损失
                bev_loss = self.criterion(map_res, map_gt, map_kernel)
                
                # 图像损失（逐视角求和）
                per_view_loss = 0.0
                for vi in range(imgs_res.shape[1]):
                    per_view_loss = per_view_loss + self.criterion(
                        imgs_res[:, vi],
                        imgs_gt[:, vi],
                        img_kernel
                    )
                per_view_loss = per_view_loss / float(imgs_res.shape[1])
                
                # 总损失
                loss = bev_loss + alpha * per_view_loss
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # 记录损失
            losses.append(loss.item())
            bev_losses.append(bev_loss.item())
            img_losses.append(per_view_loss.item())
            
            # 计算指标（不需要梯度）
            with torch.no_grad():
                # 正样本 MSE
                pooled_gt = F.adaptive_max_pool2d(map_gt, output_size=map_res.shape[-2:])
                pos_mask = pooled_gt > 0.1
                pos_mse = (
                    ((map_res - pooled_gt) ** 2)[pos_mask].mean().item()
                    if pos_mask.any()
                    else float("nan")
                )
                pos_mses.append(pos_mse)
                
                # 辅助正样本 MSE
                aux_pos_mask = imgs_gt > 0.1
                aux_pos_mse = (
                    ((imgs_res - imgs_gt) ** 2)[aux_pos_mask].mean().item()
                    if aux_pos_mask.any()
                    else float("nan")
                )
                aux_pos_mses.append(aux_pos_mse)
            
            # 定期打印日志
            if batch_idx % log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                raw_min = float(map_logits[0, 0].min().item())
                raw_max = float(map_logits[0, 0].max().item())
                mean_raw = float(map_logits[0, 0].mean().item())
                max_gt = float(map_gt[0, 0].max().item())
                
                print(
                    f"[step {self.global_step}] "
                    f"loss={loss.item():.6f} "
                    f"bev={bev_loss.item():.6f} "
                    f"img={per_view_loss.item():.6f} "
                    f"pos_mse={pos_mse:.6f} "
                    f"aux_pos_mse={aux_pos_mse:.6f} "
                    f"pred_raw=[{raw_min:.3f},{raw_max:.3f}] "
                    f"mean={mean_raw:.3f} max_gt={max_gt:.3f} "
                    f"lr={lr:.5f}"
                )
            
            self.global_step += 1
        
        # 计算 epoch 平均
        return {
            "loss": np.mean(losses),
            "bev_loss": np.mean(bev_losses),
            "img_loss": np.mean(img_losses),
            "pos_mse": np.nanmean(pos_mses),
            "aux_pos_mse": np.nanmean(aux_pos_mses),
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        map_kernel: torch.Tensor,
        img_kernel: torch.Tensor,
        alpha: float = 1.0,
    ) -> Dict[str, float]:
        """
        验证一次
        
        Args:
            val_loader: 验证数据加载器
            map_kernel: BEV 高斯核
            img_kernel: 图像高斯核
            alpha: 图像损失权重
            
        Returns:
            dict: 验证指标
        """
        self.model.eval()
        
        losses = []
        bev_losses = []
        img_losses = []
        
        with torch.no_grad():
            for stems, x_views, map_gt, imgs_gt in val_loader:
                x_views = x_views.to(self.device, non_blocking=True)
                map_gt = map_gt.to(self.device, non_blocking=True)
                imgs_gt = imgs_gt.to(self.device, non_blocking=True)
                
                map_logits, imgs_logits = self.model(x_views)
                map_res = torch.sigmoid(map_logits)
                imgs_res = torch.sigmoid(imgs_logits)
                
                bev_loss = self.criterion(map_res, map_gt, map_kernel)
                
                per_view_loss = 0.0
                for vi in range(imgs_res.shape[1]):
                    per_view_loss = per_view_loss + self.criterion(
                        imgs_res[:, vi],
                        imgs_gt[:, vi],
                        img_kernel
                    )
                per_view_loss = per_view_loss / float(imgs_res.shape[1])
                
                loss = bev_loss + alpha * per_view_loss
                
                losses.append(loss.item())
                bev_losses.append(bev_loss.item())
                img_losses.append(per_view_loss.item())
        
        return {
            "loss": np.mean(losses),
            "bev_loss": np.mean(bev_losses),
            "img_loss": np.mean(img_losses),
        }
    
    def save_checkpoint(self, epoch: int, best: bool = False):
        """
        保存模型检查点
        
        Args:
            epoch: 当前 epoch
            best: 是否为最佳模型
        """
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        
        name = "model_best.pth" if best else f"model_epoch{epoch}.pth"
        path = self.output_dir / name
        torch.save(ckpt, path)
        print(f"[CKPT] saved {path}")
    
    def save_visualizations(
        self,
        stems: list,
        map_logits: torch.Tensor,
        map_gt: torch.Tensor,
        Hb: int,
        Wb: int,
        suffix: str = "",
    ):
        """
        保存可视化热图
        
        Args:
            stems: 样本名称列表
            map_logits: BEV logits (B, 1, Hb, Wb)
            map_gt: BEV 真值 (B, 1, NBH, NBW)
            Hb: BEV 高度
            Wb: BEV 宽度
            suffix: 文件名后缀
        """
        with torch.no_grad():
            map_res = torch.sigmoid(map_logits)
            
            # 池化 GT 到 BEV 尺寸
            gt_pooled = F.adaptive_max_pool2d(map_gt, output_size=(Hb, Wb))
            
            for i, stem in enumerate(stems):
                pred = map_res[i, 0].detach().cpu().numpy()
                gt = gt_pooled[i, 0].detach().cpu().numpy()
                
                save_heat_png(self.output_dir / f"{stem}_pred{suffix}.png", pred)
                save_heat_png(self.output_dir / f"{stem}_gt{suffix}.png", gt)


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.5,
    weight_decay: float = 5e-4,
) -> torch.optim.SGD:
    """
    创建 SGD 优化器
    
    Args:
        model: 模型
        lr: 初始学习率
        momentum: 动量
        weight_decay: 权重衰减
        
    Returns:
        torch.optim.SGD: 优化器
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """
    创建 OneCycle 学习率调度器
    
    Args:
        optimizer: 优化器
        max_lr: 最大学习率
        epochs: 总 epoch 数
        steps_per_epoch: 每个 epoch 的步数
        
    Returns:
        torch.optim.lr_scheduler.OneCycleLR: 调度器
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )
