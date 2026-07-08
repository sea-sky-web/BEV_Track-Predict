"""
BEV 预测可视化：生成 GT vs Prediction 对比图 / 视频。
用法:
  单帧: python scripts/visualize_prediction.py --model_path <path> --data_root <path> --frame 0
  视频: python scripts/visualize_prediction.py --model_path <path> --data_root <path> --video
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom, scale_intrinsics
from config import (
    CAM_NAMES, DEFAULT_BACKBONE, DEFAULT_BEV_DOWN, DEFAULT_FEAT_CH,
    DEFAULT_FEAT_H, DEFAULT_FEAT_W, DEFAULT_FUSION_MODE, DEFAULT_IMG_H,
    DEFAULT_IMG_W, DEFAULT_PERSON_H, IMG_ORI_H, IMG_ORI_W,
)
from dataset import create_wildtrack_dataset
from geometry import build_mvdet_proj_mat, make_worldgrid2worldcoord_mat
from models import create_model

SCALE = 3


def load_model_and_dataset(model_path, data_root, device="cuda",
                           fusion_mode=DEFAULT_FUSION_MODE, backbone=DEFAULT_BACKBONE):
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

    sx_f = feat_hw[1] / IMG_ORI_W
    sy_f = feat_hw[0] / IMG_ORI_H
    for v in views:
        calib_cache[v]["K_feat"] = scale_intrinsics(calib_cache[v]["K0"], sx=sx_f, sy=sy_f)

    origin_x_m = float(pom.get("ORIGINE_X", -3.0))
    origin_y_m = float(pom.get("ORIGINE_Y", -9.0))
    step = (step_m * bev_down) * unit_scale
    w2w_mat = make_worldgrid2worldcoord_mat(origin_x_m * unit_scale, origin_y_m * unit_scale, step)

    proj_mats = []
    for v in views:
        proj = build_mvdet_proj_mat(calib_cache[v]["K_feat"], calib_cache[v]["R"], calib_cache[v]["t"], w2w_mat)
        proj_mats.append(torch.from_numpy(proj).float())
    proj_mats = torch.stack(proj_mats, dim=0)

    ds = create_wildtrack_dataset(
        data_root=data_root, views=views, max_frames=-1, frame_start=0,
        img_hw=(DEFAULT_IMG_H, DEFAULT_IMG_W), feat_hw=feat_hw, bev_down=bev_down,
        person_h_m=DEFAULT_PERSON_H, unit_scale=unit_scale, calib_cache=calib_cache,
    )

    dev = torch.device(device)
    model = create_model(
        num_views=len(views), proj_mats=proj_mats.to(dev), reduced_hw=(hb, wb),
        feat_hw=feat_hw, device=dev, pretrained=False, backbone=backbone,
        feat_ch=DEFAULT_FEAT_CH, add_coord=True, fusion_mode=fusion_mode,
    )

    payload = torch.load(model_path, map_location=dev, weights_only=False)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[OK] Loaded model ({len(ds)} frames, fusion={fusion_mode})")
    return model, ds, dev


def render_frame(model, ds, dev, frame_idx):
    """渲染单帧，返回 overlay 图 (H*SCALE, W*SCALE, 3) 和统计信息。"""
    stem, x_views, map_gt, imgs_gt = ds[frame_idx]
    x_views = x_views.unsqueeze(0).to(dev)

    with torch.no_grad():
        map_logits, _offset, _ = model(x_views)

    pred = map_logits[0, 0].cpu().numpy()
    gt = F.adaptive_max_pool2d(map_gt.unsqueeze(0), output_size=map_logits.shape[-2:])[0, 0].numpy()
    h, w = pred.shape

    # Raw logit 归一化
    pred_clipped = np.clip(pred, 0, None)
    pmax = pred_clipped.max()
    pred_norm = pred_clipped / pmax if pmax > 0 else pred_clipped
    pred_vis = (pred_norm * 255).clip(0, 255).astype(np.uint8)
    pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

    # GT 点
    gt_points = np.argwhere(gt > 0.5)

    # 检测峰值
    local_max = maximum_filter(pred, size=5)
    peaks = (pred == local_max) & (pred > 0.3)
    peak_points = np.argwhere(peaks)

    # Overlay: 预测热图 + GT 绿圈 + 检测红圈
    overlay = pred_color.copy()
    for r, c in gt_points:
        cv2.circle(overlay, (c, r), 5, (0, 255, 0), 2)
    for r, c in peak_points:
        cv2.circle(overlay, (c, r), 4, (0, 0, 255), 2)

    overlay_big = cv2.resize(overlay, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)

    # 添加帧信息文字
    n_gt = gt_points.shape[0]
    n_det = peak_points.shape[0]
    info = f"Frame {frame_idx}  GT:{n_gt}  Det:{n_det}  logit:[{pred.min():.2f},{pred.max():.2f}]"
    cv2.putText(overlay_big, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return overlay_big, {"n_gt": n_gt, "n_det": n_det, "logit_max": pred.max()}


def visualize_single(model, ds, dev, frame_idx, output_dir):
    """生成单帧的 prediction + overlay 图。"""
    os.makedirs(output_dir, exist_ok=True)

    stem, x_views, map_gt, imgs_gt = ds[frame_idx]
    x_views = x_views.unsqueeze(0).to(dev)

    with torch.no_grad():
        map_logits, _offset, _ = model(x_views)

    pred = map_logits[0, 0].cpu().numpy()
    gt = F.adaptive_max_pool2d(map_gt.unsqueeze(0), output_size=map_logits.shape[-2:])[0, 0].numpy()
    pred_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(pred, -20, 20)))
    h, w = pred.shape

    # GT
    gt_binary = (gt > 0.5).astype(np.uint8)
    gt_dilated = cv2.dilate(gt_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    gt_vis = (gt_dilated * 255).astype(np.uint8)
    gt_color = cv2.applyColorMap(gt_vis, cv2.COLORMAP_JET)
    gt_points = np.argwhere(gt > 0.5)
    for r, c in gt_points:
        cv2.circle(gt_color, (c, r), 5, (0, 255, 0), 2)

    # Pred
    pred_clipped = np.clip(pred, 0, None)
    pmax = pred_clipped.max()
    pred_norm = pred_clipped / pmax if pmax > 0 else pred_clipped
    pred_vis = (pred_norm * 255).clip(0, 255).astype(np.uint8)
    pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

    gt_big = cv2.resize(gt_color, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    pred_big = cv2.resize(pred_color, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    hs, ws = h * SCALE, w * SCALE

    canvas = np.zeros((hs * 2 + 60, ws, 3), dtype=np.uint8)
    canvas[0:hs, :] = gt_big
    cv2.putText(canvas, f"Ground Truth ({gt_points.shape[0]} pedestrians)", (10, hs + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    canvas[hs + 60:hs * 2 + 60, :] = pred_big
    cv2.putText(canvas, f"Prediction (logit:[{pred.min():.2f},{pred.max():.2f}])",
                (10, hs * 2 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "bev_prediction.png"), canvas)

    # Overlay
    overlay_big, _ = render_frame(model, ds, dev, frame_idx)
    cv2.imwrite(os.path.join(output_dir, "bev_overlay.png"), overlay_big)

    print(f"[OK] Saved: bev_prediction.png, bev_overlay.png")
    print(f"raw logit range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"GT > 0.5: {(gt > 0.5).sum()} pixels")


def visualize_video(model, ds, dev, output_dir, max_frames=-1, fps=5):
    """生成所有帧的 BEV 预测视频。"""
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(ds) if max_frames < 0 else min(max_frames, len(ds))
    if n_frames <= 0:
        print("[WARN] No frames to render for video.")
        return None

    print(f"[VIDEO] Rendering {n_frames} frames at {fps} fps ...")
    video_path = os.path.join(output_dir, "bev_prediction.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    try:
        for i in range(n_frames):
            frame, info = render_frame(model, ds, dev, i)
            if writer is None:
                fh, fw = frame.shape[:2]
                writer = cv2.VideoWriter(video_path, fourcc, fps, (fw, fh))
            writer.write(frame)
            if (i + 1) % 50 == 0 or i == n_frames - 1:
                print(f"  [{i+1}/{n_frames}] GT={info['n_gt']} Det={info['n_det']} logit_max={info['logit_max']:.3f}")
    finally:
        if writer is not None:
            writer.release()

    size_mb = os.path.getsize(video_path) / 1e6
    print(f"[OK] Video saved: {video_path} ({size_mb:.1f} MB, {n_frames} frames)")
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", default="outputs/visualization")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="生成所有帧的 BEV 预测视频")
    parser.add_argument("--max_frames", type=int, default=-1, help="视频最大帧数（-1=全部）")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--fusion_mode", default=DEFAULT_FUSION_MODE)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    args = parser.parse_args()

    model, ds, dev = load_model_and_dataset(
        args.model_path, args.data_root, args.device, args.fusion_mode, args.backbone)

    if args.video:
        visualize_video(model, ds, dev, args.output, args.max_frames, args.fps)
    else:
        visualize_single(model, ds, dev, args.frame, args.output)
