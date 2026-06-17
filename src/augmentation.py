"""View-level augmentation utilities for WildTrack BEV training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import math
import torch


def parse_color_jitter(raw: str | Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(raw, str):
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        vals = [float(x) for x in raw]
    if len(vals) != 4:
        raise ValueError("color_jitter must contain brightness,contrast,saturation,hue")
    if any(v < 0.0 for v in vals):
        raise ValueError(f"color_jitter values must be >= 0, got {vals}")
    if vals[3] > 0.5:
        raise ValueError(f"hue jitter must be <= 0.5, got {vals[3]}")
    return vals[0], vals[1], vals[2], vals[3]


@dataclass
class ViewCoherentAugment:
    """
    Apply per-view photometric jitter and optional frame-coherent horizontal flip.

    Horizontal flip is kept explicit because flipped camera images require matching
    feature-plane labels and should be validated with projection visualizations.
    """

    hflip_prob: float = 0.0
    color_jitter: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.05)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.hflip_prob <= 1.0:
            raise ValueError(f"hflip_prob must be in [0,1], got {self.hflip_prob}")
        self.color_jitter = parse_color_jitter(self.color_jitter)

    def _sample_factor(self, amount: float) -> float:
        if amount <= 0.0:
            return 1.0
        return random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)

    def _sample_photometric_factors(self) -> tuple[float, float, float, float]:
        brightness, contrast, saturation, hue = self.color_jitter
        hue_delta = random.uniform(-hue, hue) if hue > 0.0 else 0.0
        return (
            self._sample_factor(brightness),
            self._sample_factor(contrast),
            self._sample_factor(saturation),
            hue_delta,
        )

    def _adjust_hue(self, img: torch.Tensor, hue_delta: float) -> torch.Tensor:
        if hue_delta == 0.0:
            return img
        angle = float(hue_delta) * 2.0 * math.pi
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        mat = img.new_tensor(
            [
                [0.299 + 0.701 * cos_a + 0.168 * sin_a, 0.587 - 0.587 * cos_a + 0.330 * sin_a, 0.114 - 0.114 * cos_a - 0.497 * sin_a],
                [0.299 - 0.299 * cos_a - 0.328 * sin_a, 0.587 + 0.413 * cos_a + 0.035 * sin_a, 0.114 - 0.114 * cos_a + 0.292 * sin_a],
                [0.299 - 0.300 * cos_a + 1.250 * sin_a, 0.587 - 0.588 * cos_a - 1.050 * sin_a, 0.114 + 0.886 * cos_a - 0.203 * sin_a],
            ]
        )
        flat = img.reshape(3, -1)
        return (mat @ flat).reshape_as(img).clamp(0.0, 1.0)

    def _photometric(self, img: torch.Tensor, factors: tuple[float, float, float, float]) -> torch.Tensor:
        brightness, contrast, saturation, hue_delta = factors
        out = img
        out = (out * brightness).clamp(0.0, 1.0)
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = ((out - mean) * contrast + mean).clamp(0.0, 1.0)
        gray = out.mean(dim=0, keepdim=True)
        out = ((out - gray) * saturation + gray).clamp(0.0, 1.0)
        return self._adjust_hue(out, hue_delta)

    def __call__(
        self,
        imgs: torch.Tensor,
        map_gt: torch.Tensor,
        aux_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return imgs, map_gt, aux_gt

        factors = self._sample_photometric_factors()
        aug_imgs = torch.stack([self._photometric(img, factors) for img in imgs], dim=0)
        if random.random() < self.hflip_prob:
            aug_imgs = torch.flip(aug_imgs, dims=(-1,))
            map_gt = torch.flip(map_gt, dims=(-1,))
            aux_gt = torch.flip(aux_gt, dims=(-1,))
        return aug_imgs, map_gt, aux_gt
