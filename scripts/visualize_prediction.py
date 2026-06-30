"""
BEV 预测可视化：生成 GT vs Prediction 对比图。
用法: python scripts/visualize_prediction.py --model_path <path> --data_root <path> --output <path>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom, scale_intrinsics
from config import (
    CAM_NAMES,
    DEFAULT_BACKBONE,
    DEFAULT_BEV_DOWN,
    DEFAULT_FEAT_CH,
    DEFAULT_FEAT_H,
    DEFAULT_FEAT_W,
    DEFAULT_FUSION_MODE,
    DEFAULT_IMG_H,
    DEFAULT_IMG_W,
    DEFAULT_PERSON_H,
    IMG_ORI_H,
    IMG_ORI_W,
)
from dataset import create_wildtrack_dataset
from geometry import build_mvdet_proj_mat, compute_valid_ratio_from_homography, make_worldgrid2worldcoord_mat
from models import create_model


def visualize(model_path, data_root, output_dir, device="cuda", frame_idx=0,
              fusion_mode=DEFAULT_FUSION_MODE, backbone=DEFAULT_BACKBONE):
    os.makedirs(output_dir, exist_ok=True)
    from pathlib import Path

    data_root = Path(data_root)
    views = list(range(7))
    feat_hw = (DEFAULT_FEAT_H, DEFAULT_FEAT_W)
    bev_down = DEFAULT_BEV_DOWN

    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    calib_loader = CalibrationLoader(data_root / "calibrations", CAM_NAMES)
    calib_cache, t_norms = calib_loader.load_all(views)

    step_m = float(pom.get("STEP", 0.025))
    unit_scale = decide_unit_scale(step_m, t_norms)

    hb = int(pom.get("NB_HEIGHT", 480)) // bev_down
    wb = int(pom.get("NB_WIDTH", 1440)) // bev_down
    hf, wf = feat_hw

    sx_f = wf / IMG_ORI_W
    sy_f = hf / IMG_ORI_H
    for v in views:
        calib_cache[v]["K_feat"] = scale_intrinsics(calib_cache[v]["K0"], sx=sx_f, sy=sy_f)

    origin_x_m = float(pom.get("ORIGINE_X", -3.0))
    origin_y_m = float(pom.get("ORIGINE_Y", -9.0))
    step = (step_m * bev_down) * unit_scale
    ox = origin_x_m * unit_scale
    oy = origin_y_m * unit_scale
    w2w_mat = make_worldgrid2worldcoord_mat(ox, oy, step)

    proj_mats = []
    for v in views:
        proj = build_mvdet_proj_mat(calib_cache[v]["K_feat"], calib_cache[v]["R"], calib_cache[v]["t"], w2w_mat)
        proj_mats.append(torch.from_numpy(proj).float())
    proj_mats = torch.stack(proj_mats, dim=0)

    ds = create_wildtrack_dataset(
        data_root=data_root,
        views=views,
        max_frames=-1,
        frame_start=0,
        img_hw=(DEFAULT_IMG_H, DEFAULT_IMG_W),
        feat_hw=feat_hw,
        bev_down=bev_down,
        person_h_m=DEFAULT_PERSON_H,
        unit_scale=unit_scale,
        calib_cache=calib_cache,
    )

    dev = torch.device(device)
    model = create_model(
        num_views=len(views),
        proj_mats=proj_mats.to(dev),
        reduced_hw=(hb, wb),
        feat_hw=feat_hw,
        device=dev,
        pretrained=False,
        backbone=backbone,
        feat_ch=DEFAULT_FEAT_CH,
        add_coord=True,
        fusion_mode=fusion_mode,
    )

    payload = torch.load(model_path, map_location=dev, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[OK] Loaded model from {model_path} (fusion={fusion_mode}, backbone={backbone})")

    stem, x_views, map_gt, imgs_gt = ds[frame_idx]
    x_views = x_views.unsqueeze(0).to(dev)

    with torch.no_grad():
        map_logits, _ = model(x_views)

    pred = map_logits[0, 0].cpu().numpy()
    gt = F.adaptive_max_pool2d(map_gt.unsqueeze(0), output_size=map_logits.shape[-2:])[0, 0].numpy()

    pred_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(pred, -20, 20)))

    # --- Normalize for visibility ---
    # GT: binary dots are tiny on 120x360, dilate to make visible
    gt_binary = (gt > 0.5).astype(np.uint8)
    gt_dilated = cv2.dilate(gt_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    gt_vis = (gt_dilated * 255).astype(np.uint8)

    # Pred: normalize to full [0,255] range for contrast
    pred_norm = pred_sigmoid.copy()
    pmax = pred_norm.max()
    if pmax > 0:
        pred_norm = pred_norm / pmax
    pred_vis = (pred_norm * 255).clip(0, 255).astype(np.uint8)

    pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
    gt_color = cv2.applyColorMap(gt_vis, cv2.COLORMAP_JET)

    # Draw green circles on GT for each pedestrian center
    gt_points = np.argwhere(gt > 0.5)  # (row, col)
    for r, c in gt_points:
        cv2.circle(gt_color, (c, r), 5, (0, 255, 0), 2)

    h, w = pred_vis.shape
    # Scale up for visibility (120x360 is very small)
    scale = 3
    gt_big = cv2.resize(gt_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    pred_big = cv2.resize(pred_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    hs, ws = h * scale, w * scale

    canvas = np.zeros((hs * 2 + 60, ws, 3), dtype=np.uint8)
    canvas[0:hs, :] = gt_big
    cv2.putText(canvas, f"Ground Truth ({gt_points.shape[0]} pedestrians)", (10, hs + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    canvas[hs + 60:hs * 2 + 60, :] = pred_big
    cv2.putText(canvas, f"Prediction (logit:[{pred.min():.2f},{pred.max():.2f}] sig>{0.3:.1f}:{(pred_sigmoid > 0.3).sum()}px)",
                (10, hs * 2 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = os.path.join(output_dir, "bev_prediction.png")
    cv2.imwrite(out_path, canvas)
    print(f"[OK] Saved: {out_path}")

    # --- Overlay: prediction heatmap + GT circles ---
    overlay = pred_color.copy()
    thresh = 0.3
    # Mark GT positions with green circles
    for r, c in gt_points:
        cv2.circle(overlay, (c, r), 5, (0, 255, 0), 2)
    # Mark detections (pred > thresh) with red circles
    det_points = np.argwhere(pred_sigmoid > thresh)
    # Use NMS-style: find local maxima
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(pred_sigmoid, size=5)
    peaks = (pred_sigmoid == local_max) & (pred_sigmoid > thresh)
    peak_points = np.argwhere(peaks)
    for r, c in peak_points:
        cv2.circle(overlay, (c, r), 4, (0, 0, 255), 2)

    overlay_big = cv2.resize(overlay, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    overlay_path = os.path.join(output_dir, "bev_overlay.png")
    cv2.imwrite(overlay_path, overlay_big)
    print(f"[OK] Saved: {overlay_path}")

    print(f"\n=== Prediction Stats ===")
    print(f"raw logit range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"sigmoid mean:    {pred_sigmoid.mean():.6f}")
    print(f"sigmoid > 0.3:   {(pred_sigmoid > 0.3).sum()} pixels")
    print(f"sigmoid > 0.5:   {(pred_sigmoid > 0.5).sum()} pixels")
    print(f"GT > 0.5:        {(gt > 0.5).sum()} pixels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", default="outputs/visualization")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--fusion_mode", default=DEFAULT_FUSION_MODE)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    args = parser.parse_args()
    visualize(args.model_path, args.data_root, args.output, args.device, args.frame,
              args.fusion_mode, args.backbone)
