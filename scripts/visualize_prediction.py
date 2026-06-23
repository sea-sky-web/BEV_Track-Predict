"""
在训练完成后生成 BEV 预测可视化图。
用法: python scripts/visualize_prediction.py --model_path <path> --data_root <path> --output <path>
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from src.config import *
from src.model import MultiviewDetector
from src.dataset import WildtrackDataset
from src.loss import GaussianMSE
from src.utils import gaussian_kernel


def visualize(model_path, data_root, output_dir, device='cuda', frame_idx=0):
    os.makedirs(output_dir, exist_ok=True)

    dataset = WildtrackDataset(
        root=data_root,
        views=[0, 1, 2, 3, 4, 5, 6],
        img_h=DEFAULT_IMG_H, img_w=DEFAULT_IMG_W,
        feat_h=DEFAULT_FEAT_H, feat_w=DEFAULT_FEAT_W,
        max_frames=-1
    )

    model = MultiviewDetector(
        backbone=DEFAULT_BACKBONE,
        pretrained=False,
        feat_ch=DEFAULT_FEAT_CH,
        bev_w=NB_WIDTH // DEFAULT_BEV_DOWN,
        bev_h=NB_HEIGHT // DEFAULT_BEV_DOWN,
        fusion_mode=DEFAULT_FUSION_MODE,
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()

    sample = dataset[frame_idx]
    imgs = sample['images'].unsqueeze(0).to(device)
    map_gt = sample['map_gt']
    projs = sample['proj_mats'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(imgs, projs)
        if isinstance(output, dict):
            map_logits = output['map_logits']
        elif isinstance(output, (tuple, list)):
            map_logits = output[0]
        else:
            map_logits = output

    pred = torch.sigmoid(map_logits).squeeze().cpu().numpy()
    gt = map_gt.squeeze().cpu().numpy()

    pred_vis = (pred * 255).clip(0, 255).astype(np.uint8)
    gt_vis = (gt * 255).clip(0, 255).astype(np.uint8)

    pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
    gt_color = cv2.applyColorMap(gt_vis, cv2.COLORMAP_JET)

    h, w = pred_vis.shape
    canvas = np.zeros((h * 2 + 40, w, 3), dtype=np.uint8)

    canvas[0:h, :] = gt_color
    cv2.putText(canvas, 'Ground Truth', (10, h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    canvas[h + 40:h * 2 + 40, :] = pred_color
    cv2.putText(canvas, f'Prediction (max={pred.max():.3f}, mean={pred.mean():.4f})',
                (10, h * 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path = os.path.join(output_dir, 'bev_prediction.png')
    cv2.imwrite(out_path, canvas)
    print(f"[OK] Saved: {out_path}")

    overlay = gt_color.copy()
    thresh = 0.3
    mask = pred > thresh
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * pred_color[mask]).astype(np.uint8)

    for y in range(gt.shape[0]):
        for x in range(gt.shape[1]):
            if gt[y, x] > 0.5:
                cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)

    pred_binary = (pred > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), 2)

    overlay_path = os.path.join(output_dir, 'bev_overlay.png')
    cv2.imwrite(overlay_path, overlay)
    print(f"[OK] Saved: {overlay_path}")

    print(f"\n=== Prediction Stats ===")
    print(f"pred max:  {pred.max():.4f}")
    print(f"pred mean: {pred.mean():.6f}")
    print(f"pred > 0.3 pixels: {(pred > 0.3).sum()}")
    print(f"pred > 0.5 pixels: {(pred > 0.5).sum()}")
    print(f"GT > 0.5 pixels:   {(gt > 0.5).sum()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output', default='outputs/visualization')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frame', type=int, default=0)
    args = parser.parse_args()
    visualize(args.model_path, args.data_root, args.output, args.device, args.frame)
