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
    # __file__ 在 colab exec (IPython kernel) 中不可用，用显式路径
    repo_dir = Path("/content/BEV_Track-Predict")
    src_dir = str(repo_dir / "src")
    if not Path(src_dir).exists():
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

    # ── E. 模型架构对比 ───────────────────────────────────────
    banner("E. 模型架构对比 (参数量 & 层结构)")

    # MVDet 模型结构
    mvdet_model = mvdet["model"]
    ours_model = ours["model"]

    # img_classifier / img_head 对比
    print("\n  --- img_classifier (MVDet) vs img_head (Ours) ---")
    if hasattr(mvdet_model, 'img_classifier'):
        for name, layer in mvdet_model.img_classifier.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                print(f"  MVDet img_classifier.{name}: {layer}")
    if hasattr(ours_model, 'img_head'):
        for name, layer in ours_model.img_head.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                print(f"  Ours  img_head.{name}: {layer}")

    # map_classifier / bev_head 对比
    print("\n  --- map_classifier (MVDet) vs bev_head (Ours) ---")
    if hasattr(mvdet_model, 'map_classifier'):
        for name, layer in mvdet_model.map_classifier.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                print(f"  MVDet map_classifier.{name}: {layer}")
    if hasattr(ours_model, 'bev_head'):
        for name, layer in ours_model.bev_head.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                print(f"  Ours  bev_head.{name}: {layer}")

    # 参数量对比
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    print(f"\n  MVDet 总参数量: {count_params(mvdet_model):,}")
    print(f"  Ours  总参数量: {count_params(ours_model):,}")

    # backbone 对比
    print("\n  --- backbone 参数量 ---")
    if hasattr(mvdet_model, 'base'):
        mvdet_bb_params = count_params(mvdet_model.base)
        print(f"  MVDet backbone (base): {mvdet_bb_params:,}")
    if hasattr(ours_model, 'backbone'):
        ours_bb_params = count_params(ours_model.backbone)
        print(f"  Ours  backbone:        {ours_bb_params:,}")

    # ── F. Forward pass 端到端对比 ──────────────────────────
    banner("F. Forward pass 端到端对比 (随机初始化，对比输出量级)")

    # 把两个模型都移到同一设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mvdet_model = mvdet_model.to(device).eval()
    ours_model = ours_model.to(device).eval()

    # 手动将 MVDet 的非 buffer 属性移到对应设备
    if hasattr(mvdet_model, 'proj_mats') and isinstance(mvdet_model.proj_mats, list):
        mvdet_model.proj_mats = [m.to(device) for m in mvdet_model.proj_mats]
    if hasattr(mvdet_model, 'coord_map') and isinstance(mvdet_model.coord_map, torch.Tensor):
        mvdet_model.coord_map = mvdet_model.coord_map.to(device)

    # 准备输入
    mvdet_imgs = mvdet["imgs"].unsqueeze(0).to(device)  # (1, V, C, H, W) or (1, V*C, H, W)
    ours_imgs = ours["x_views"].unsqueeze(0).to(device)  # (1, V, C, H, W)

    with torch.no_grad():
        # MVDet forward
        try:
            mvdet_out = mvdet_model(mvdet_imgs)
            if isinstance(mvdet_out, (tuple, list)):
                mvdet_map = mvdet_out[0]
                mvdet_img_out = mvdet_out[1] if len(mvdet_out) > 1 else None
            else:
                mvdet_map = mvdet_out
                mvdet_img_out = None
            print(f"  MVDet map output: shape={tuple(mvdet_map.shape)}, "
                  f"min={mvdet_map.min().item():.6f}, max={mvdet_map.max().item():.6f}, "
                  f"mean={mvdet_map.mean().item():.6f}, std={mvdet_map.std().item():.6f}")
            if mvdet_img_out is not None:
                print(f"  MVDet img output: shape={tuple(mvdet_img_out.shape)}, "
                      f"min={mvdet_img_out.min().item():.6f}, max={mvdet_img_out.max().item():.6f}")
        except Exception as e:
            print(f"  [ERROR] MVDet forward failed: {e}")
            mvdet_map = None

        # Ours forward
        try:
            ours_map, ours_img_out = ours_model(ours_imgs)
            print(f"  Ours  map output: shape={tuple(ours_map.shape)}, "
                  f"min={ours_map.min().item():.6f}, max={ours_map.max().item():.6f}, "
                  f"mean={ours_map.mean().item():.6f}, std={ours_map.std().item():.6f}")
            print(f"  Ours  img output: shape={tuple(ours_img_out.shape)}, "
                  f"min={ours_img_out.min().item():.6f}, max={ours_img_out.max().item():.6f}")
        except Exception as e:
            print(f"  [ERROR] Ours forward failed: {e}")
            ours_map = None

    # shape 对比
    if mvdet_map is not None and ours_map is not None:
        if mvdet_map.shape == ours_map.shape:
            print(f"\n  [PASS] map output shape 一致: {tuple(mvdet_map.shape)}")
        else:
            print(f"\n  [FAIL] map output shape 不一致: MVDet={tuple(mvdet_map.shape)} vs Ours={tuple(ours_map.shape)}")
            all_pass = False
    else:
        print("\n  [FAIL] 因模型 forward 失败，无法对比 map output shape")
        all_pass = False

    # ── G. 逐阶段 feature 对比（使用 hook）────────────────────
    banner("G. 逐阶段 feature stats (backbone → warp → concat → head)")

    # MVDet 内部 feature 提取
    print("\n  --- MVDet 内部 features ---")
    try:
        mvdet_model.eval()
        with torch.no_grad():
            # MVDet 的 forward 内部：
            # 1) base(imgs) -> feat, 2) img_classifier(feat), 3) warp, 4) concat+coord, 5) map_classifier
            B, _, C, H, W = mvdet_imgs.shape if mvdet_imgs.dim() == 5 else (1, 7, 3, 720, 1280)
            
            # 尝试提取中间特征
            if hasattr(mvdet_model, 'base'):
                # 取第一个视角的 backbone 输出
                single_img = mvdet_imgs.reshape(-1, C, H, W)[:1]  # 第一个视角
                bb_feat = mvdet_model.base(single_img)
                print(f"  MVDet backbone output: shape={tuple(bb_feat.shape)}, "
                      f"mean={bb_feat.mean().item():.6f}, std={bb_feat.std().item():.6f}")
    except Exception as e:
        print(f"  [WARN] MVDet feature extraction failed: {e}")

    print("\n  --- Ours 内部 features ---")
    try:
        ours_model.eval()
        with torch.no_grad():
            single_img = ours_imgs[:, 0]  # (1, 3, H, W)
            bb_feat = ours_model.backbone(single_img)
            bb_feat_interp = torch.nn.functional.interpolate(
                bb_feat, size=(ours_model.Hf, ours_model.Wf),
                mode="bilinear", align_corners=False)
            print(f"  Ours  backbone output: shape={tuple(bb_feat.shape)}, "
                  f"mean={bb_feat.mean().item():.6f}, std={bb_feat.std().item():.6f}")
            print(f"  Ours  backbone (interp): shape={tuple(bb_feat_interp.shape)}")

            # Warp to BEV
            from geometry import warp_perspective_torch
            M = ours_model.proj_mats[0].unsqueeze(0)
            bev_feat = warp_perspective_torch(bb_feat_interp, M,
                                              dsize=(ours_model.Hb, ours_model.Wb))
            print(f"  Ours  warp BEV (view 0): shape={tuple(bev_feat.shape)}, "
                  f"mean={bev_feat.mean().item():.6f}, std={bev_feat.std().item():.6f}, "
                  f"nonzero_ratio={((bev_feat.abs() > 1e-6).float().mean().item()):.4f}")
    except Exception as e:
        print(f"  [WARN] Ours feature extraction failed: {e}")

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
