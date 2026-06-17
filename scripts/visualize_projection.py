"""Visualize WildTrack BEV-to-image projection coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom, scale_intrinsics
from config import (
    CAM_NAMES,
    DEFAULT_BEV_DOWN,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEAT_H,
    DEFAULT_FEAT_W,
    DEFAULT_IMG_H,
    DEFAULT_IMG_W,
    DEFAULT_VIEWS,
    IMG_ORI_H,
    IMG_ORI_W,
)
from geometry import build_mvdet_proj_mat, compute_valid_ratio_from_homography, make_worldgrid2worldcoord_mat


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Save projection overlays for WildTrack views.")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--views", type=str, default=DEFAULT_VIEWS)
    ap.add_argument("--frame_idx", type=int, default=0)
    ap.add_argument("--img_h", type=int, default=DEFAULT_IMG_H)
    ap.add_argument("--img_w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--feat_h", type=int, default=DEFAULT_FEAT_H)
    ap.add_argument("--feat_w", type=int, default=DEFAULT_FEAT_W)
    ap.add_argument("--bev_down", type=int, default=DEFAULT_BEV_DOWN)
    ap.add_argument("--grid_stride", type=int, default=20)
    ap.add_argument("--output_dir", type=str, default="docs/geometry_validation")
    return ap.parse_args()


def _parse_views(raw: str) -> list[int]:
    views = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    if not views:
        raise ValueError("At least one view id is required.")
    return views


def _find_image(data_root: Path, view: int, stem: str) -> Path:
    img_dir = data_root / "Image_subsets" / f"C{view + 1}"
    for ext in (".png", ".jpg", ".jpeg"):
        path = img_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for stem={stem} in {img_dir}")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted((data_root / "annotations_positions").glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotations found under {data_root / 'annotations_positions'}")
    if args.frame_idx < 0 or args.frame_idx >= len(ann_files):
        raise ValueError(f"frame_idx={args.frame_idx} out of range for {len(ann_files)} annotations")
    stem = ann_files[args.frame_idx].stem

    views = _parse_views(args.views)
    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    calib_cache, t_norms = CalibrationLoader(data_root / "calibrations", CAM_NAMES).load_all(views)
    step_m = float(pom.get("STEP", 0.025))
    unit_scale = decide_unit_scale(step_m, t_norms)
    hb = int(pom.get("NB_HEIGHT", 1440)) // args.bev_down
    wb = int(pom.get("NB_WIDTH", 480)) // args.bev_down

    sx_f = args.feat_w / IMG_ORI_W
    sy_f = args.feat_h / IMG_ORI_H
    for view in views:
        calib_cache[view]["K_feat"] = scale_intrinsics(calib_cache[view]["K0"], sx=sx_f, sy=sy_f)

    origin_x_m = float(pom.get("ORIGINE_X", -3.0))
    origin_y_m = float(pom.get("ORIGINE_Y", -9.0))
    step = step_m * args.bev_down * unit_scale
    w2w = make_worldgrid2worldcoord_mat(origin_x_m * unit_scale, origin_y_m * unit_scale, step)

    summary: list[dict[str, object]] = []
    grid_x = np.arange(0, wb, max(1, args.grid_stride), dtype=np.float64)
    grid_y = np.arange(0, hb, max(1, args.grid_stride), dtype=np.float64)
    yy, xx = np.meshgrid(grid_y, grid_x, indexing="ij")
    bev_pts = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(xx.size)], axis=0)

    for view in views:
        calib = calib_cache[view]
        proj = build_mvdet_proj_mat(calib["K_feat"], calib["R"], calib["t"], w2w)
        valid_ratio = compute_valid_ratio_from_homography(proj, (args.feat_h, args.feat_w), (hb, wb))
        feat_pts = np.linalg.inv(proj) @ bev_pts
        feat_x = feat_pts[0] / (feat_pts[2] + 1e-9)
        feat_y = feat_pts[1] / (feat_pts[2] + 1e-9)
        valid = (
            np.isfinite(feat_x)
            & np.isfinite(feat_y)
            & (feat_x >= 0)
            & (feat_x <= args.feat_w - 1)
            & (feat_y >= 0)
            & (feat_y <= args.feat_h - 1)
        )

        image = Image.open(_find_image(data_root, view, stem)).convert("RGB").resize((args.img_w, args.img_h))
        draw = ImageDraw.Draw(image)
        sx_img = args.img_w / args.feat_w
        sy_img = args.img_h / args.feat_h
        for x, y in zip(feat_x[valid], feat_y[valid]):
            px = float(x * sx_img)
            py = float(y * sy_img)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(40, 220, 120), outline=(0, 80, 40))

        out_path = out_dir / f"projection_view{view}_{stem}.png"
        image.save(out_path)
        summary.append(
            {
                "view": view,
                "camera": calib["cam"],
                "stem": stem,
                "valid_ratio": float(valid_ratio),
                "overlay": str(out_path),
            }
        )

    summary_path = out_dir / f"projection_summary_{stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
