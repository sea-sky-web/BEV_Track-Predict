# src/trainer.py
"""
训练器模块：训练循环、验证、日志记录和检查点管理
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import create_loss_criterion
from utils import save_heat_png


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
        freeze_backbone_epochs: int = 0,
        bev_pos_weight: float = 1.0,
        bev_neg_weight: float = 1.0,
        img_pos_weight: float = 1.0,
        img_neg_weight: float = 1.0,
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
            freeze_backbone_epochs: 前多少个 epoch 冻结 backbone
            bev_pos_weight: BEV 热图正样本损失权重
            bev_neg_weight: BEV 热图负样本损失权重
            img_pos_weight: 图像热图正样本损失权重
            img_neg_weight: 图像热图负样本损失权重
        """
        for name, value in {
            "bev_pos_weight": bev_pos_weight,
            "bev_neg_weight": bev_neg_weight,
            "img_pos_weight": img_pos_weight,
            "img_neg_weight": img_neg_weight,
        }.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 损失函数：默认权重 1.0 保持原 GaussianMSE 行为。
        self.bev_criterion = create_loss_criterion(
            weighted=(bev_pos_weight != 1.0 or bev_neg_weight != 1.0),
            pos_weight=bev_pos_weight,
            neg_weight=bev_neg_weight,
        )
        self.img_criterion = create_loss_criterion(
            weighted=(img_pos_weight != 1.0 or img_neg_weight != 1.0),
            pos_weight=img_pos_weight,
            neg_weight=img_neg_weight,
        )
        self.criterion = self.bev_criterion
        
        # AMP 梯度缩放器
        self.scaler = torch.amp.GradScaler(
            device=str(device),
            enabled=(amp_enabled and device.type == "cuda")
        )
        self.amp_enabled = amp_enabled and device.type == "cuda"
        self.freeze_bn = freeze_bn
        if freeze_backbone_epochs < 0:
            raise ValueError(f"freeze_backbone_epochs must be >= 0, got {freeze_backbone_epochs}")
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self._backbone_is_frozen = False
        
        # 冻结 BatchNorm
        if self.freeze_bn:
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

    def _set_backbone_trainable(self, trainable: bool) -> None:
        """Toggle the shared image encoder while keeping optimizer groups stable."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return
        for p in backbone.parameters():
            p.requires_grad_(trainable)
        if trainable:
            backbone.train()
        else:
            backbone.eval()
        self._backbone_is_frozen = not trainable

    def _apply_backbone_freeze(self, epoch: int) -> None:
        should_freeze = epoch < self.freeze_backbone_epochs
        changed = should_freeze != self._backbone_is_frozen
        if should_freeze:
            # model.train() is called at each epoch, so re-apply eval/grad state.
            self._set_backbone_trainable(False)
        elif changed:
            self._set_backbone_trainable(not should_freeze)
        if changed:
            state = "frozen" if should_freeze else "trainable"
            print(f"[TRAIN] backbone={state} epoch={epoch} freeze_backbone_epochs={self.freeze_backbone_epochs}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        map_kernel: torch.Tensor,
        img_kernel: torch.Tensor,
        epoch: int = 0,
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
                - "raw_pos_mse": 正样本区域 raw logits vs smoothed GT 的 MSE
                - "raw_neg_mse": 背景区域 raw logits vs smoothed GT 的 MSE
                - "snr": 信噪比（正样本 logit 均值 - 背景 logit 均值）
        """
        self.model.train()
        self._apply_backbone_freeze(epoch)
        # model.train() 会把 BN 重新切回训练态，这里再次冻结确保语义稳定
        if self.freeze_bn:
            self._freeze_bn()
        
        losses = []
        bev_losses = []
        img_losses = []
        raw_pos_mses = []
        raw_neg_mses = []
        snrs = []
        
        for batch_idx, (stems, x_views, map_gt, imgs_gt) in enumerate(train_loader):
            x_views = x_views.to(self.device, non_blocking=True)
            map_gt = map_gt.to(self.device, non_blocking=True)
            imgs_gt = imgs_gt.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                map_logits, imgs_logits = self.model(x_views)
                
                # MVDet: loss 直接在 raw logits 上计算，不经过 sigmoid
                bev_loss = self.bev_criterion(map_logits, map_gt, map_kernel)
                
                per_view_loss = 0.0
                for vi in range(imgs_logits.shape[1]):
                    per_view_loss = per_view_loss + self.img_criterion(
                        imgs_logits[:, vi],
                        imgs_gt[:, vi],
                        img_kernel
                    )
                per_view_loss = per_view_loss / float(imgs_logits.shape[1])
                
                loss = bev_loss + alpha * per_view_loss
            
            if not torch.isfinite(loss):
                if batch_idx % log_every == 0:
                    raw_min = float(torch.nan_to_num(map_logits[0, 0], nan=0.0).min().item())
                    raw_max = float(torch.nan_to_num(map_logits[0, 0], nan=0.0).max().item())
                    print(
                        f"[step {self.global_step}] non-finite loss detected, skip update "
                        f"(bev={bev_loss.item():.6f}, img={per_view_loss.item():.6f}, "
                        f"pred_raw=[{raw_min:.3f},{raw_max:.3f}])"
                    )
                self.global_step += 1
                continue

            # 反向传播
            if self.amp_enabled:
                scale_before = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # 仅在 optimizer 真正更新后推进 LR
                if self.scaler.get_scale() >= scale_before:
                    self.scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
            
            # 记录损失
            losses.append(loss.item())
            bev_losses.append(bev_loss.item())
            img_losses.append(per_view_loss.item())
            
            # 计算指标（raw logits vs Gaussian-smoothed GT，与 loss 同空间）
            with torch.no_grad():
                B, C, H, W = map_logits.shape
                pooled_gt = F.adaptive_max_pool2d(map_gt, output_size=(H, W))

                # 构建 Gaussian-smoothed GT（复现 loss 内部的目标）
                _tgt = pooled_gt.reshape(B * C, 1, H, W)
                _k = map_kernel.to(dtype=_tgt.dtype, device=_tgt.device)
                _pad = (_k.shape[-1] - 1) // 2
                smoothed_gt = F.conv2d(_tgt, _k, padding=_pad).reshape(B, C, H, W)

                pos_mask = smoothed_gt > 0.1
                neg_mask = smoothed_gt < 0.01
                diff2 = (map_logits - smoothed_gt) ** 2

                raw_pos_mse = diff2[pos_mask].mean().item() if pos_mask.any() else float("nan")
                raw_neg_mse = diff2[neg_mask].mean().item() if neg_mask.any() else float("nan")
                snr = (
                    (map_logits[pos_mask].mean() - map_logits[neg_mask].mean()).item()
                    if pos_mask.any() and neg_mask.any()
                    else float("nan")
                )
                raw_pos_mses.append(raw_pos_mse)
                raw_neg_mses.append(raw_neg_mse)
                snrs.append(snr)
            
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
                    f"raw_pos_mse={raw_pos_mse:.6f} "
                    f"raw_neg_mse={raw_neg_mse:.6f} "
                    f"snr={snr:.3f} "
                    f"pred_raw=[{raw_min:.3f},{raw_max:.3f}] "
                    f"mean={mean_raw:.3f} max_gt={max_gt:.3f} "
                    f"lr={lr:.5f}"
                )
            
            self.global_step += 1
        
        # 计算 epoch 平均
        if not losses:
            return {
                "loss": float("nan"),
                "bev_loss": float("nan"),
                "img_loss": float("nan"),
                "raw_pos_mse": float("nan"),
                "raw_neg_mse": float("nan"),
                "snr": float("nan"),
            }

        return {
            "loss": np.mean(losses),
            "bev_loss": np.mean(bev_losses),
            "img_loss": np.mean(img_losses),
            "raw_pos_mse": np.nanmean(raw_pos_mses),
            "raw_neg_mse": np.nanmean(raw_neg_mses),
            "snr": np.nanmean(snrs),
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
        raw_pos_mses = []
        raw_neg_mses = []
        snrs = []
        
        with torch.no_grad():
            for stems, x_views, map_gt, imgs_gt in val_loader:
                x_views = x_views.to(self.device, non_blocking=True)
                map_gt = map_gt.to(self.device, non_blocking=True)
                imgs_gt = imgs_gt.to(self.device, non_blocking=True)
                
                map_logits, imgs_logits = self.model(x_views)
                
                bev_loss = self.bev_criterion(map_logits, map_gt, map_kernel)
                
                per_view_loss = 0.0
                for vi in range(imgs_logits.shape[1]):
                    per_view_loss = per_view_loss + self.img_criterion(
                        imgs_logits[:, vi],
                        imgs_gt[:, vi],
                        img_kernel
                    )
                per_view_loss = per_view_loss / float(imgs_logits.shape[1])
                
                loss = bev_loss + alpha * per_view_loss
                
                losses.append(loss.item())
                bev_losses.append(bev_loss.item())
                img_losses.append(per_view_loss.item())

                # 与 train_epoch 一致的检测指标
                B, C, H, W = map_logits.shape
                pooled_gt = F.adaptive_max_pool2d(map_gt, output_size=(H, W))
                _tgt = pooled_gt.reshape(B * C, 1, H, W)
                _k = map_kernel.to(dtype=_tgt.dtype, device=_tgt.device)
                _pad = (_k.shape[-1] - 1) // 2
                smoothed_gt = F.conv2d(_tgt, _k, padding=_pad).reshape(B, C, H, W)

                pos_mask = smoothed_gt > 0.1
                neg_mask = smoothed_gt < 0.01
                diff2 = (map_logits - smoothed_gt) ** 2

                raw_pos_mses.append(diff2[pos_mask].mean().item() if pos_mask.any() else float("nan"))
                raw_neg_mses.append(diff2[neg_mask].mean().item() if neg_mask.any() else float("nan"))
                snrs.append(
                    (map_logits[pos_mask].mean() - map_logits[neg_mask].mean()).item()
                    if pos_mask.any() and neg_mask.any()
                    else float("nan")
                )
        
        return {
            "loss": np.mean(losses),
            "bev_loss": np.mean(bev_losses),
            "img_loss": np.mean(img_losses),
            "raw_pos_mse": np.nanmean(raw_pos_mses),
            "raw_neg_mse": np.nanmean(raw_neg_mses),
            "snr": np.nanmean(snrs),
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
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    momentum: float = 0.5,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    创建优化器，并给 backbone 使用 0.1x 学习率。
    
    Args:
        model: 模型
        optimizer_name: adam 或 sgd
        lr: 初始学习率
        momentum: 动量
        weight_decay: 权重衰减
        
    Returns:
        torch.optim.Optimizer: 优化器
    """
    optimizer_name = optimizer_name.lower()
    backbone = getattr(model, "backbone", None)
    backbone_ids = set()
    param_groups = []

    if backbone is not None:
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone.parameters()}
        if backbone_params:
            # MVDet uses same lr for all params; only reduce backbone lr for Adam
            bb_lr = lr * 0.1 if optimizer_name == "adam" else lr
            param_groups.append({"name": "backbone", "params": backbone_params, "lr": bb_lr})

    head_params = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]
    if head_params:
        param_groups.append({"name": "head", "params": head_params, "lr": lr})

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer")

    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    max_lr: float,
    epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    创建学习率调度器。默认 cosine，OneCycle 保留 legacy 复现路径。
    
    Args:
        optimizer: 优化器
        scheduler_name: cosine 或 onecycle
        max_lr: 最大学习率
        epochs: 总 epoch 数
        steps_per_epoch: 每个 epoch 的步数
        
    Returns:
        torch.optim.lr_scheduler.LRScheduler: 调度器
    """
    scheduler_name = scheduler_name.lower()
    total_steps = max(int(epochs) * int(steps_per_epoch), 1)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )
    if scheduler_name == "onecycle":
        max_lrs = []
        for group in optimizer.param_groups:
            if group.get("name") == "backbone":
                # SGD: same lr for all (MVDet style); Adam: reduce backbone lr
                is_sgd = isinstance(optimizer, torch.optim.SGD)
                max_lrs.append(max_lr if is_sgd else max_lr * 0.1)
            else:
                max_lrs.append(max_lr)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
