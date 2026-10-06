#!/usr/bin/env python3
"""Verify WildTrack positionID, camera projection, and BEV homography alignment.

This is a read-only sanity check. It does not train a model or write outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

DEFAULT_BEV_DOWN = 4
DEFAULT_FEAT_H = 270
DEFAULT_FEAT_W = 480
DEFAULT_IMG_ORI_H = 1080
DEFAULT_IMG_ORI_W = 1920
DEFAULT_NB_HEIGHT = 1440
DEFAULT_NB_WIDTH = 480
DEFAULT_ORIGINE_X_M = -3.0
DEFAULT_ORIGINE_Y_M = -9.0
DEFAULT_STEP_M = 0.025


def load_project_modules() -> None:
    global np
    global CalibrationLoader, decide_unit_scale, parse_rectangles_pom, scale_intrinsics
    global CAM_NAMES, IMG_ORI_H, IMG_ORI_W, NB_HEIGHT, NB_WIDTH, ORIGINE_X_M, ORIGINE_Y_M, STEP_M
    global build_mvdet_proj_mat, compute_valid_ratio_from_homography, make_worldgrid2worldcoord_mat

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import numpy as _np
    from calibration import CalibrationLoader as _CalibrationLoader
    from calibration import decide_unit_scale as _decide_unit_scale
    from calibration import parse_rectangles_pom as _parse_rectangles_pom
    from calibration import scale_intrinsics as _scale_intrinsics
    from config import CAM_NAMES as _CAM_NAMES
    from config import IMG_ORI_H as _IMG_ORI_H
    from config import IMG_ORI_W as _IMG_ORI_W
    from config import NB_HEIGHT as _NB_HEIGHT
    from config import NB_WIDTH as _NB_WIDTH
    from config import ORIGINE_X_M as _ORIGINE_X_M
    from config import ORIGINE_Y_M as _ORIGINE_Y_M
    from config import STEP_M as _STEP_M
    from geometry import build_mvdet_proj_mat as _build_mvdet_proj_mat
    from geometry import compute_valid_ratio_from_homography as _compute_valid_ratio_from_homography
    from geometry import make_worldgrid2worldcoord_mat as _make_worldgrid2worldcoord_mat

    np = _np
    CalibrationLoader = _CalibrationLoader
    decide_unit_scale = _decide_unit_scale
    parse_rectangles_pom = _parse_rectangles_pom
    scale_intrinsics = _scale_intrinsics
    CAM_NAMES = _CAM_NAMES
    IMG_ORI_H = _IMG_ORI_H
    IMG_ORI_W = _IMG_ORI_W
    NB_HEIGHT = _NB_HEIGHT
    NB_WIDTH = _NB_WIDTH
    ORIGINE_X_M = _ORIGINE_X_M
    ORIGINE_Y_M = _ORIGINE_Y_M
    STEP_M = _STEP_M
    build_mvdet_proj_mat = _build_mvdet_proj_mat
    compute_valid_ratio_from_homography = _compute_valid_ratio_from_homography
    make_worldgrid2worldcoord_mat = _make_worldgrid2worldcoord_mat


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Check positionID -> world -> feature -> reduced BEV homography closure."
    )
    ap.add_argument("--data_root", type=str, default="wildtrack", help="WildTrack dataset root")
    ap.add_argument("--views", type=str, default="0,1,2", help="Comma-separated view IDs")
    ap.add_argument("--max_frames", type=int, default=5, help="Number of annotation files to inspect")
    ap.add_argument("--max_points", type=int, default=50, help="Max annotation points to inspect")
    ap.add_argument("--bev_down", type=int, default=DEFAULT_BEV_DOWN, help="Reduced BEV stride")
    ap.add_argument("--feat_h", type=int, default=DEFAULT_FEAT_H, help="Feature map height")
    ap.add_argument("--feat_w", type=int, default=DEFAULT_FEAT_W, help="Feature map width")
    ap.add_argument("--mean_thr", type=float, default=1.0, help="Mean closure error threshold in BEV cells")
    ap.add_argument("--p95_thr", type=float, default=2.0, help="P95 closure error threshold in BEV cells")
    return ap.parse_args()


def parse_views(raw: str) -> List[int]:
    views = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not views:
        raise ValueError("At least one view is required.")
    return views


def require_dataset_root(data_root: Path) -> None:
    required = [
        data_root / "annotations_positions",
        data_root / "calibrations",
        data_root / "Image_subsets",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "WildTrack dataset root is incomplete. Missing: " + ", ".join(missing)
        )


def load_annotation_points(data_root: Path, nb_w: int, max_frames: int, max_points: int) -> List[Tuple[str, int, int]]:
    ann_dir = data_root / "annotations_positions"
    ann_files = sorted(ann_dir.glob("*.json"))
    if max_frames > 0:
        ann_files = ann_files[:max_frames]

    points: List[Tuple[str, int, int]] = []
    for ann_path in ann_files:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        for obj in data:
            pos_id = obj.get("positionID")
            if pos_id is None:
                continue
            pos_id = int(pos_id)
            ix = pos_id % nb_w
            iy = pos_id // nb_w
            points.append((ann_path.stem, ix, iy))
            if max_points > 0 and len(points) >= max_points:
                return points
    return points


def build_projection_context(
    data_root: Path,
    views: Iterable[int],
    feat_hw: Tuple[int, int],
    bev_down: int,
) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray], float, Tuple[int, int], Dict[str, float], float]:
    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    nb_h = int(pom.get("NB_HEIGHT", DEFAULT_NB_HEIGHT))
    nb_w = int(pom.get("NB_WIDTH", DEFAULT_NB_WIDTH))
    hb = nb_h // bev_down
    wb = nb_w // bev_down

    calib_loader = CalibrationLoader(data_root / "calibrations", CAM_NAMES)
    calib_cache, t_norms = calib_loader.load_all(list(views))

    step_m = float(pom.get("STEP", STEP_M))
    unit_scale = decide_unit_scale(step_m, t_norms)
    print(f"[UNIT] step={step_m}, median||t||={np.median(t_norms):.2f} => unit_scale={unit_scale}")

    hf, wf = feat_hw
    sx_f = wf / IMG_ORI_W
    sy_f = hf / IMG_ORI_H
    for v in views:
        calib_cache[v]["K_feat"] = scale_intrinsics(calib_cache[v]["K0"], sx=sx_f, sy=sy_f)

    origin_x_m = float(pom.get("ORIGINE_X", ORIGINE_X_M))
    origin_y_m = float(pom.get("ORIGINE_Y", ORIGINE_Y_M))
    reduced_step = (step_m * bev_down) * unit_scale
    ox = origin_x_m * unit_scale
    oy = origin_y_m * unit_scale
    w2w_mat = make_worldgrid2worldcoord_mat(ox, oy, reduced_step)

    proj_mats: Dict[int, np.ndarray] = {}
    for v in views:
        calib = calib_cache[v]
        proj = build_mvdet_proj_mat(calib["K_feat"], calib["R"], calib["t"], w2w_mat)
        vr = compute_valid_ratio_from_homography(proj, (hf, wf), (hb, wb))
        print(f"[GRID] view={v} cam={calib['cam']} valid_ratio={vr:.4f}")
        proj_mats[v] = proj

    grid_info = {
        "nb_h": float(nb_h),
        "nb_w": float(nb_w),
        "origin_x": ox,
        "origin_y": oy,
        "full_step": step_m * unit_scale,
        "reduced_step": reduced_step,
    }
    return calib_cache, proj_mats, unit_scale, (hb, wb), grid_info, step_m


def project_world_to_feature(calib: Dict, world_point: np.ndarray) -> Tuple[float, float, float]:
    pc = calib["R"] @ world_point.reshape(3, 1) + calib["t"]
    z = float(pc[2, 0])
    if z <= 1e-9:
        return float("nan"), float("nan"), z
    k = calib["K_feat"]
    u = float(k[0, 0] * (pc[0, 0] / z) + k[0, 2])
    v = float(k[1, 1] * (pc[1, 0] / z) + k[1, 2])
    return u, v, z


def apply_homography(mat: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    dst = mat @ np.array([x, y, 1.0], dtype=np.float64)
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    views = parse_views(args.views)
    try:
        require_dataset_root(data_root)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 2
    load_project_modules()

    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    nb_w = int(pom.get("NB_WIDTH", NB_WIDTH))
    points = load_annotation_points(data_root, nb_w=nb_w, max_frames=args.max_frames, max_points=args.max_points)
    if not points:
        print("[FAIL] checked_points=0; no annotation positionID values were found.")
        return 2

    calib_cache, proj_mats, _, reduced_hw, grid_info, _ = build_projection_context(
        data_root=data_root,
        views=views,
        feat_hw=(args.feat_h, args.feat_w),
        bev_down=args.bev_down,
    )
    hb, wb = reduced_hw
    print(f"[CFG] feat={args.feat_h}x{args.feat_w}, bev(reduced)={hb}x{wb}, views={views}")

    errors: List[float] = []
    skipped_behind = 0
    skipped_outside = 0
    checked = 0

    ox = grid_info["origin_x"]
    oy = grid_info["origin_y"]
    full_step = grid_info["full_step"]
    reduced_step = grid_info["reduced_step"]

    for stem, ix, iy in points:
        xw = ox + (ix + 0.5) * full_step
        yw = oy + (iy + 0.5) * full_step
        expected_x = (xw - ox) / reduced_step - 0.5
        expected_y = (yw - oy) / reduced_step - 0.5
        world_foot = np.array([xw, yw, 0.0], dtype=np.float64)

        for view in views:
            u, v, z = project_world_to_feature(calib_cache[view], world_foot)
            if z <= 1e-9:
                skipped_behind += 1
                continue
            if not (0.0 <= u <= args.feat_w - 1 and 0.0 <= v <= args.feat_h - 1):
                skipped_outside += 1
                continue

            actual_x, actual_y = apply_homography(proj_mats[view], u, v)
            err = float(np.hypot(actual_x - expected_x, actual_y - expected_y))
            errors.append(err)
            checked += 1

            if checked <= 5:
                print(
                    "[SAMPLE] "
                    f"frame={stem} view={view} full=({ix},{iy}) "
                    f"expected_reduced=({expected_x:.3f},{expected_y:.3f}) "
                    f"actual_reduced=({actual_x:.3f},{actual_y:.3f}) "
                    f"err={err:.6f}"
                )

    if checked == 0:
        print(
            "[FAIL] checked_points=0; all candidate footpoints were behind cameras "
            "or outside selected feature planes."
        )
        print(f"[COUNTS] annotation_points={len(points)} skipped_behind={skipped_behind} skipped_outside={skipped_outside}")
        return 2

    arr = np.array(errors, dtype=np.float64)
    mean_err = float(arr.mean())
    p95_err = float(np.percentile(arr, 95))
    max_err = float(arr.max())
    status = "PASS" if mean_err <= args.mean_thr and p95_err <= args.p95_thr else "FAIL"

    print(
        "[SUMMARY] "
        f"status={status} checked_points={checked} annotation_points={len(points)} "
        f"skipped_behind={skipped_behind} skipped_outside={skipped_outside} "
        f"mean_reprojection_cell_error={mean_err:.6f} "
        f"p95_reprojection_cell_error={p95_err:.6f} "
        f"max_reprojection_cell_error={max_err:.6f}"
    )

    if status != "PASS":
        print(f"[FAIL] thresholds mean<={args.mean_thr} p95<={args.p95_thr}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
