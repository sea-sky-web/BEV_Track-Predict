"""Loss functions for spatiotemporal field prediction.

L_total = L_occ + lambda_vel * L_vel + lambda_trace * L_trace

L_occ:   0.5 * weighted_BCE + 0.5 * soft_Dice
L_vel:   occupied-mask SmoothL1
L_trace: advection consistency (predicted velocity advects predicted occupancy)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupancyLoss(nn.Module):

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        if valid_mask is not None:
            mask = valid_mask.float()
        else:
            mask = torch.ones_like(pred)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * mask).sum() / mask.sum().clamp(min=1.0)

        pred_m = pred * mask
        target_m = target * mask
        intersection = (pred_m * target_m).sum()
        union = pred_m.sum() + target_m.sum()
        dice = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)

        return self.bce_weight * bce + self.dice_weight * dice


class VelocityLoss(nn.Module):

    def forward(
        self,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        gt_vx: torch.Tensor,
        gt_vy: torch.Tensor,
        occ_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = (occ_mask > 0.5).float()
        n_occupied = mask.sum().clamp(min=1.0)

        loss_vx = F.smooth_l1_loss(pred_vx * mask, gt_vx * mask, reduction="sum")
        loss_vy = F.smooth_l1_loss(pred_vy * mask, gt_vy * mask, reduction="sum")

        return (loss_vx + loss_vy) / n_occupied


class TraceConsistencyLoss(nn.Module):
    """Advection consistency: predicted velocity should advect predicted occupancy
    to match the next-step predicted occupancy."""

    def __init__(self, dt: float = 0.5, cell_m: float = 0.1):
        super().__init__()
        self.dt = dt
        self.cell_m = cell_m

    def _advect(self, occ: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
        b, h, w = occ.shape
        row_grid = torch.arange(h, device=occ.device, dtype=occ.dtype).view(1, h, 1).expand(b, h, w)
        col_grid = torch.arange(w, device=occ.device, dtype=occ.dtype).view(1, 1, w).expand(b, h, w)

        src_row = row_grid - vx * self.dt / self.cell_m
        src_col = col_grid - vy * self.dt / self.cell_m

        grid_y = 2.0 * src_row / (h - 1) - 1.0
        grid_x = 2.0 * src_col / (w - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        occ_4d = occ.unsqueeze(1)
        advected = F.grid_sample(occ_4d, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return advected.squeeze(1)

    def forward(
        self,
        pred_occ_steps: list[torch.Tensor],
        pred_vx_steps: list[torch.Tensor],
        pred_vy_steps: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred_occ_steps[0].device, dtype=pred_occ_steps[0].dtype)
        n_pairs = 0

        for t in range(len(pred_occ_steps) - 1):
            advected = self._advect(pred_occ_steps[t], pred_vx_steps[t], pred_vy_steps[t])
            target = pred_occ_steps[t + 1]
            loss = loss + F.mse_loss(advected, target)
            n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs
        return loss


class CombinedTemporalLoss(nn.Module):

    def __init__(
        self,
        lambda_vel: float = 0.5,
        lambda_trace: float = 0.1,
        ablation: str = "full",
        dt: float = 0.5,
        cell_m: float = 0.1,
    ):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_trace = lambda_trace
        self.ablation = ablation

        self.occ_loss = OccupancyLoss()
        self.vel_loss = VelocityLoss()
        self.trace_loss = TraceConsistencyLoss(dt=dt, cell_m=cell_m)

    def forward(
        self,
        pred: torch.Tensor,
        gt_fields: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: (B, T_future, 3, H, W) — [occ_logit, vx, vy]
            gt_fields: (B, T_future, 5, H, W) — [occ, vx, vy, conf, valid]
            valid_mask: optional (B, H, W) or (B, T_future, H, W)

        Returns:
            dict with 'total', 'occ', 'vel', 'trace' losses
        """
        b, t_fut, _, h, w = pred.shape

        gt_occ = gt_fields[:, :, 0]
        gt_vx = gt_fields[:, :, 1]
        gt_vy = gt_fields[:, :, 2]

        if valid_mask is None:
            if gt_fields.shape[2] >= 5:
                vm = gt_fields[:, :, 4]
            else:
                vm = None
        else:
            vm = valid_mask

        total_occ = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        total_vel = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for t in range(t_fut):
            step_vm = vm[:, t] if vm is not None and vm.ndim == 4 else vm
            total_occ = total_occ + self.occ_loss(pred[:, t, 0], gt_occ[:, t], step_vm)

            if self.ablation in ("occ_vel", "full"):
                total_vel = total_vel + self.vel_loss(
                    pred[:, t, 1], pred[:, t, 2],
                    gt_vx[:, t], gt_vy[:, t],
                    gt_occ[:, t],
                )

        total_occ = total_occ / t_fut
        total_vel = total_vel / t_fut if self.ablation in ("occ_vel", "full") else total_vel

        total_trace = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.ablation == "full" and t_fut > 1:
            pred_occ_sigmoid = [torch.sigmoid(pred[:, t, 0]) for t in range(t_fut)]
            pred_vx_list = [pred[:, t, 1] for t in range(t_fut)]
            pred_vy_list = [pred[:, t, 2] for t in range(t_fut)]
            total_trace = self.trace_loss(pred_occ_sigmoid, pred_vx_list, pred_vy_list)

        total = total_occ + self.lambda_vel * total_vel + self.lambda_trace * total_trace

        return {
            "total": total,
            "occ": total_occ.detach(),
            "vel": total_vel.detach(),
            "trace": total_trace.detach(),
        }
