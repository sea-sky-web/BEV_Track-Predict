"""Save confidence_v2 per-view fusion weight maps for one WildTrack frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (
    DEFAULT_BACKBONE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BEV_DOWN,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEAT_CH,
    DEFAULT_FEAT_H,
    DEFAULT_FEAT_W,
    DEFAULT_FUSION_MODE,
    DEFAULT_IMG_H,
    DEFAULT_IMG_W,
    DEFAULT_PERSON_H,
    DEFAULT_VALID_THR,
    DEFAULT_VIEWS,
)
from dataset import create_wildtrack_dataset
from evaluate_main import _build_projection, _load_model_weights
from models import create_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize per-view confidence fusion weights.")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--views", type=str, default=DEFAULT_VIEWS)
    ap.add_argument("--frame_idx", type=int, default=0)
    ap.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, choices=["resnet18", "resnet50"])
    ap.add_argument("--fusion_mode", type=str, default=DEFAULT_FUSION_MODE, choices=["confidence", "confidence_v2"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--img_h", type=int, default=DEFAULT_IMG_H)
    ap.add_argument("--img_w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--feat_h", type=int, default=DEFAULT_FEAT_H)
    ap.add_argument("--feat_w", type=int, default=DEFAULT_FEAT_W)
    ap.add_argument("--bev_down", type=int, default=DEFAULT_BEV_DOWN)
    ap.add_argument("--person_h", type=float, default=DEFAULT_PERSON_H)
    ap.add_argument("--valid_thr", type=float, default=DEFAULT_VALID_THR)
    ap.add_argument("--output_dir", type=str, default="docs/fusion_weights")
    return ap.parse_args()


def _parse_views(raw: str) -> list[int]:
    views = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    if not views:
        raise ValueError("At least one view id is required.")
    return views


def main() -> int:
    args = parse_args()
    dev = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    views = _parse_views(args.views)
    feat_hw = (args.feat_h, args.feat_w)
    calib_cache, proj_mats, kept_views, unit_scale, reduced_hw, _, _ = _build_projection(
        data_root=data_root,
        views=views,
        feat_hw=feat_hw,
        bev_down=args.bev_down,
        drop_bad_views=False,
        valid_thr=args.valid_thr,
    )
    dataset = create_wildtrack_dataset(
        data_root=data_root,
        views=kept_views,
        max_frames=1,
        frame_start=args.frame_idx,
        img_hw=(args.img_h, args.img_w),
        feat_hw=feat_hw,
        bev_down=args.bev_down,
        person_h_m=args.person_h,
        unit_scale=unit_scale,
        calib_cache=calib_cache,
    )

    def collate_fn(batch):
        stems, x_views, map_gt, imgs_gt = zip(*batch)
        return list(stems), torch.stack(x_views, 0), torch.stack(map_gt, 0), torch.stack(imgs_gt, 0)

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)
    stems, x_views, _, _ = next(iter(loader))
    model = create_model(
        num_views=len(kept_views),
        proj_mats=proj_mats.to(dev),
        reduced_hw=reduced_hw,
        feat_hw=feat_hw,
        device=dev,
        pretrained=False,
        backbone=args.backbone,
        feat_ch=DEFAULT_FEAT_CH,
        add_coord=True,
        fusion_mode=args.fusion_mode,
    )
    _load_model_weights(model, Path(args.model_path), dev)
    model.eval()
    with torch.no_grad():
        model(x_views.to(dev))

    weights = getattr(getattr(model, "confidence_fusion", None), "latest_weights", None)
    if weights is None:
        raise RuntimeError("No fusion weights captured; use --fusion_mode confidence_v2 or confidence alias.")
    weights = weights[0].detach().cpu()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, view in enumerate(kept_views):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(weights[idx], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"view {view} fusion weight")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out_path = out_dir / f"fusion_weight_view{view}_{stems[0]}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"[OK] wrote fusion weights to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
