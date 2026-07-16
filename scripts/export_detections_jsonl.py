"""Export frozen detector BEV detections to JSONL with world coordinates.

Usage (requires checkpoint + GPU):
    python scripts/export_detections_jsonl.py \
        --data_root wildtrack \
        --model_path outputs/frozen_detector/model_final.pth \
        --output detections.jsonl \
        --device cuda

Dry-run (no checkpoint):
    python scripts/export_detections_jsonl.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from config import (
    CAM_NAMES,
    DEFAULT_BEV_DOWN,
    DEFAULT_FEAT_CH,
    DEFAULT_FEAT_H,
    DEFAULT_FEAT_W,
    DEFAULT_IMG_H,
    DEFAULT_IMG_W,
    NB_HEIGHT,
    NB_WIDTH,
    ORIGINE_X_M,
    ORIGINE_Y_M,
    STEP_M,
)


def _reduced_to_world(row: float, col: float, bev_down: int = DEFAULT_BEV_DOWN) -> tuple[float, float]:
    cell_m = STEP_M * bev_down
    world_x_m = ORIGINE_X_M + (row + 0.5) * cell_m
    world_y_m = ORIGINE_Y_M + (col + 0.5) * cell_m
    return world_x_m, world_y_m


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen detector detections to JSONL")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="detections.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--views", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    parser.add_argument("--fusion_mode", type=str, default="geo_confidence_v1")
    parser.add_argument("--threshold", type=float, default=0.375)
    parser.add_argument("--nms_radius", type=float, default=5.0)
    parser.add_argument("--frame_start", type=int, default=360)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--bev_down", type=int, default=DEFAULT_BEV_DOWN)
    parser.add_argument("--feat_h", type=int, default=DEFAULT_FEAT_H)
    parser.add_argument("--feat_w", type=int, default=DEFAULT_FEAT_W)
    parser.add_argument("--img_h", type=int, default=DEFAULT_IMG_H)
    parser.add_argument("--img_w", type=int, default=DEFAULT_IMG_W)
    parser.add_argument("--feat_ch", type=int, default=DEFAULT_FEAT_CH)

    args = parser.parse_args()

    from evaluate_main import _build_projection, _extract_points, _load_model_weights
    from models import create_model
    from dataset import create_wildtrack_dataset
    from torch.utils.data import DataLoader

    data_root = Path(args.data_root)
    device = torch.device(args.device)
    views = [int(v) for v in args.views.split(",")]
    feat_hw = (args.feat_h, args.feat_w)

    calib_cache, proj_mats, kept_views, unit_scale, bev_hw, pom, step_m = _build_projection(
        data_root, views, feat_hw, args.bev_down, drop_bad_views=False, valid_thr=0.0
    )

    model = create_model(
        backbone=args.backbone,
        fusion_mode=args.fusion_mode,
        proj_mats=proj_mats,
        feat_hw=feat_hw,
        reduced_hw=bev_hw,
        num_views=len(kept_views),
        device=device,
        pretrained=False,
        feat_ch=args.feat_ch,
        add_coord=True,
    )
    _load_model_weights(model, Path(args.model_path), device)
    model.eval()

    ds = create_wildtrack_dataset(
        data_root=data_root,
        views=kept_views,
        calib_cache=calib_cache,
        max_frames=args.max_frames,
        frame_start=args.frame_start,
        img_h=args.img_h,
        img_w=args.img_w,
        feat_h=args.feat_h,
        feat_w=args.feat_w,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    out_path = Path(args.output)
    n_total = 0
    with open(out_path, "w") as f, torch.no_grad():
        for batch_idx, (stems, x_views, _map_gt, _imgs_gt) in enumerate(loader):
            x_views = x_views.to(device, non_blocking=True)
            map_logits, offset_preds, _ = model(x_views)

            for bi in range(map_logits.shape[0]):
                frame_idx = args.frame_start + batch_idx * args.batch + bi
                stem = stems[bi] if isinstance(stems, (list, tuple)) else stems

                pts = _extract_points(
                    map_logits[bi, 0].cpu(),
                    threshold=args.threshold,
                    nms_ksize=3,
                    max_preds=0,
                    min_distance=args.nms_radius,
                )

                for pi in range(pts.shape[0]):
                    row, col, score = float(pts[pi, 0]), float(pts[pi, 1]), float(pts[pi, 2])
                    wx, wy = _reduced_to_world(row, col, args.bev_down)
                    record = {
                        "frame_index": frame_idx,
                        "frame_stem": str(stem),
                        "world_x_m": round(wx, 6),
                        "world_y_m": round(wy, 6),
                        "score": round(score, 6),
                        "row_reduced": round(row, 2),
                        "col_reduced": round(col, 2),
                    }
                    f.write(json.dumps(record) + "\n")
                    n_total += 1

            if batch_idx % 10 == 0:
                print(f"[export] frame {frame_idx}, detections so far: {n_total}")

    print(f"Exported {n_total} detections to {out_path}")


if __name__ == "__main__":
    main()
