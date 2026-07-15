"""Trainer for spatiotemporal field prediction (ConvLSTM).

Follows MVDetTrainer patterns with early stopping and seed management.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from temporal.convlstm import SpatioTemporalPredictor
from temporal.temporal_loss import CombinedTemporalLoss
from temporal.field_metrics import compute_occupancy_auprc


class TemporalTrainer:

    def __init__(
        self,
        model: SpatioTemporalPredictor,
        criterion: CombinedTemporalLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        output_dir: Path,
        patience: int = 10,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience

        self.best_auprc = -1.0
        self.epochs_without_improvement = 0
        self.global_step = 0

    def train_epoch(self, loader: DataLoader, epoch: int, log_every: int = 10) -> dict[str, float]:
        self.model.train()
        losses = {"total": [], "occ": [], "vel": [], "trace": []}

        for batch_idx, (history, future) in enumerate(loader):
            history = history.to(self.device, non_blocking=True)
            future = future.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(history)
            loss_dict = self.criterion(pred, future)

            loss = loss_dict["total"]
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at epoch {epoch} step {batch_idx}, skipping")
                continue

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1

            for k in losses:
                losses[k].append(loss_dict[k].item())

            if batch_idx % log_every == 0:
                print(
                    f"[train e{epoch} s{batch_idx}] "
                    f"total={loss.item():.6f} occ={loss_dict['occ'].item():.6f} "
                    f"vel={loss_dict['vel'].item():.6f} trace={loss_dict['trace'].item():.6f}"
                )

        return {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        losses = {"total": [], "occ": [], "vel": [], "trace": []}
        all_auprc = []

        for history, future in loader:
            history = history.to(self.device, non_blocking=True)
            future = future.to(self.device, non_blocking=True)

            pred = self.model(history)
            loss_dict = self.criterion(pred, future)

            for k in losses:
                losses[k].append(loss_dict[k].item())

            pred_occ = torch.sigmoid(pred[:, :, 0]).cpu().numpy()
            gt_occ = future[:, :, 0].cpu().numpy()

            for b in range(pred_occ.shape[0]):
                for t in range(pred_occ.shape[1]):
                    auprc = compute_occupancy_auprc(pred_occ[b, t], gt_occ[b, t])
                    all_auprc.append(auprc)

        metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}
        metrics["auprc"] = float(np.mean(all_auprc)) if all_auprc else 0.0
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.output_dir / ("best_model.pth" if best else "latest_model.pth")
        torch.save(ckpt, path)

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def check_early_stop(self, val_auprc: float) -> bool:
        if val_auprc > self.best_auprc:
            self.best_auprc = val_auprc
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
