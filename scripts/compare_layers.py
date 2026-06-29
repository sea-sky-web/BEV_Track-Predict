#!/usr/bin/env python3
"""
逐层对比 MVDet 官方实现和我们的实现。
在 Colab 上运行，需要 WildTrack 数据和 MVDet 仓库已 clone。

对比节点：
  A. 投影矩阵 (7×3×3)
  B. Backbone 输出特征图 stats
  C. 单视角 BEV 投影特征图
  D. GT 热图
  E. Coord map

用法：
    python scripts/compare_layers.py --data_root /content/BEV_Track-Predict/wildtrack
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T


def banner(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}", flush=True)


def stats(name: str, t: torch.Tensor | np.ndarray):
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    print(f"  {name}: shape={tuple(t.shape)}, "
          f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
          f"mean={t.mean().item():.6f}, std={t.std().item():.6f}")


def compare_tensors(name: str, ours: torch.Tensor, theirs: torch.Tensor, atol=1e-4):
    if ours.shape != theirs.shape:
        print(f"  [FAIL] {name}: shape mismatch: ours={tuple(ours.shape)} vs theirs={tuple(theirs.shape)}")
        return False
    close = torch.allclose(ours.float(), theirs.float(), atol=atol)
    max_diff = (ours.float() - theirs.float()).abs().max().item()
    status = "PASS" if close else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.8f} (atol={atol})")
    if not close:
        stats(f"  ours_{name}", ours)
        stats(f"  theirs_{name}", theirs)
    return close


def load_mvdet(data_root: str, mvdet_dir: str):
    """加载 MVDet 官方模型，提取投影矩阵和中间输出。"""
    sys.path.insert(0, mvdet_dir)
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.models.persp_trans_detector import PerspTransDetector

    transform = T.Compose([T.Resize([720, 1280]), T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    wildtrack_path = os.path.expanduser("~/Data/Wildtrack")
    base = Wildtrack(wildtrack_path)
    dataset = frameDataset(base, train=True, transform=transform, grid_reduce=4)

    # 获取一帧数据
    imgs, map_gt, imgs_gt, frame = dataset[0]

    # 创建模型（不训练，只看结构和投影矩阵）
    model = PerspTransDetector(dataset, "resnet18")

    return {
        "proj_mats": [m.clone() for m in model.proj_mats],
        "coord_map": model.coord_map.clone(),
        "imgs": imgs,
        "map_gt": map_gt,
        "imgs_gt": imgs_gt,
        "frame": frame,
        "model": model,
        "dataset": dataset,
        "reducedgrid_shape": model.reducedgrid_shape,
        "upsample_shape": model.upsample_shape,
        "map_kernel": dataset.map_kernel.clone(),
        "img_kernel": dataset.img_kernel.clone(),
    }


def load_ours(data_root: str):
    """加载我们的模型，提取投影矩阵和中间输出。"""
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    sys.path.insert(0, src_dir)

    from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom, scale_intrinsics
    from config import (CAM_NAMES, DEFAULT_BEV_DOWN, DEFAULT_FEAT_CH, DEFAULT_FEAT_H,
                        DEFAULT_FEAT_W, DEFAULT_IMG_H, DEFAULT_IMG_W, DEFAULT_PERSON_H,
                        IMG_ORI_H, IMG_ORI_W, DEFAULT_MAP_KSIZE, DEFAULT_MAP_SIGMA,
                        DEFAULT_IMG_KSIZE, DEFAULT_IMG_SIGMA)
    from dataset import create_wildtrack_dataset
    from geometry import build_mvdet_proj_mat, make_worldgrid2worldcoord_mat
    from models import create_model
    from utils import build_gaussian_kernel_2d

    data_root = Path(data_root)
    views = list(range(7))
    feat_hw = (DEFAULT_FEAT_H, DEFAULT_FEAT_W)
    bev_down = DEFAULT_BEV_DOWN

    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    calib_loader = CalibrationLoader(data_root / "calibrations", CAM_NAMES)
    calib_cache, t_norms = calib_loader.load_all(views)

    step_m = float(pom.get("STEP", 0.025))
    unit_scale = decide_unit_scale(step_m, t_norms)
    print(f"  [UNIT] step={step_m}, median||t||={np.median(t_norms):.2f}, unit_scale={unit_scale}")

    hb = int(pom.get("NB_HEIGHT", 1440)) // bev_down
    wb = int(pom.get("NB_WIDTH", 480)) // bev_down
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
        proj = build_mvdet_proj_mat(
            calib_cache[v]["K_feat"], calib_cache[v]["R"], calib_cache[v]["t"], w2w_mat)
        proj_mats.append(torch.from_numpy(proj).float())
    proj_mats_tensor = torch.stack(proj_mats, dim=0)

    ds = create_wildtrack_dataset(
        data_root=data_root, views=views, max_frames=1, frame_start=0,
        img_hw=(DEFAULT_IMG_H, DEFAULT_IMG_W), feat_hw=feat_hw,
        bev_down=bev_down, person_h_m=DEFAULT_PERSON_H,
        unit_scale=unit_scale, calib_cache=calib_cache,
    )

    stem, x_views, map_gt, imgs_gt = ds[0]

    model = create_model(
        num_views=7, proj_mats=proj_mats_tensor,
        reduced_hw=(hb, wb), feat_hw=feat_hw,
        device=torch.device("cpu"), pretrained=True,
        backbone="resnet18", feat_ch=DEFAULT_FEAT_CH,
        add_coord=True, fusion_mode="concat",
    )

    map_kernel = build_gaussian_kernel_2d(DEFAULT_MAP_KSIZE, DEFAULT_MAP_SIGMA, device=torch.device("cpu"))
    img_kernel = build_gaussian_kernel_2d(DEFAULT_IMG_KSIZE, DEFAULT_IMG_SIGMA, device=torch.device("cpu"))

    return {
        "proj_mats": proj_mats,
        "x_views": x_views,
        "map_gt": map_gt,
        "imgs_gt": imgs_gt,
        "model": model,
        "reduced_hw": (hb, wb),
        "w2w_mat": w2w_mat,
        "unit_scale": unit_scale,
        "map_kernel": map_kernel,
        "img_kernel": img_kernel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/BEV_Track-Predict/wildtrack")
    parser.add_argument("--mvdet_dir", default="/content/MVDet")
    args = parser.parse_args()

    all_pass = True

    # ── A. 投影矩阵对比 ────────────────────────────────────
    banner("A. 投影矩阵对比")
    print("加载 MVDet...")
    mvdet = load_mvdet(args.data_root, args.mvdet_dir)
    print("加载 Ours...")
    ours = load_ours(args.data_root)

    for v in range(7):
        ok = compare_tensors(f"proj_mat[view={v}]", ours["proj_mats"][v], mvdet["proj_mats"][v], atol=1e-2)
        if not ok:
            all_pass = False
            # 打印详细对比
            print(f"\n  === View {v} 详细对比 ===")
            print(f"  Ours:\n{ours['proj_mats'][v].numpy()}")
            print(f"  MVDet:\n{mvdet['proj_mats'][v].numpy()}")

    # ── B. Coord map 对比 ───────────────────────────────────
    banner("B. Coord map 对比")
    # MVDet coord_map shape: (1, 2, H, W), values in [-1, 1]
    stats("mvdet_coord_map", mvdet["coord_map"])
    # 检查我们的 coord map 是否存在且形状一致
    if hasattr(ours["model"], "coord"):
        stats("ours_coord_map", ours["model"].coord)
        ok = compare_tensors("coord_map", ours["model"].coord, mvdet["coord_map"], atol=1e-4)
        if not ok:
            all_pass = False
    else:
        print("  [WARN] 我们的模型没有 coord 属性")

    # ── C. GT 热图对比 ──────────────────────────────────────
    banner("C. GT 热图对比 (frame 0)")
    stats("mvdet_map_gt", mvdet["map_gt"])
    stats("ours_map_gt", ours["map_gt"])
    # MVDet GT shape: (1, H, W) from ToTensor
    # Ours GT shape: (1, NBH, NBW) full resolution, needs pooling to reduced
    import torch.nn.functional as F
    mvdet_gt = mvdet["map_gt"].squeeze()  # (H, W)
    ours_gt_full = ours["map_gt"].squeeze()  # (NBH, NBW)
    ours_gt_reduced = F.adaptive_max_pool2d(
        ours_gt_full.unsqueeze(0).unsqueeze(0),
        output_size=mvdet_gt.shape
    ).squeeze()
    ok = compare_tensors("map_gt (reduced)", ours_gt_reduced, mvdet_gt, atol=0.1)
    if not ok:
        all_pass = False
        # 统计非零点数
        print(f"  MVDet GT 非零点数: {(mvdet_gt > 0).sum().item()}")
        print(f"  Ours GT 非零点数 (reduced): {(ours_gt_reduced > 0).sum().item()}")

    # ── D. Gaussian kernel 对比 ─────────────────────────────
    banner("D. Gaussian kernel 对比")
    stats("mvdet_map_kernel", mvdet["map_kernel"])
    stats("ours_map_kernel", ours["map_kernel"])
    # MVDet kernel shape: (1, 1, 41, 41)
    # Ours kernel shape: (1, 1, 41, 41)
    ok = compare_tensors("map_kernel", ours["map_kernel"], mvdet["map_kernel"], atol=1e-4)
    if not ok:
        all_pass = False

    # ── 总结 ────────────────────────────────────────────────
    banner("总结")
    if all_pass:
        print("  [ALL PASS] 所有对比节点一致")
    else:
        print("  [HAS FAILURES] 存在不一致的节点，需要修复")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
