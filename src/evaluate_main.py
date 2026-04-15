"""
评估脚本：使用与 train_main 一致的数据与投影链路，计算离线损失指标。
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom, scale_intrinsics
from config import (
    CAM_NAMES,
    DEFAULT_ALPHA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BEV_DOWN,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEAT_CH,
    DEFAULT_FEAT_H,
    DEFAULT_FEAT_W,
    DEFAULT_IMG_H,
    DEFAULT_IMG_KSIZE,
    DEFAULT_IMG_SIGMA,
    DEFAULT_IMG_W,
    DEFAULT_MAX_FRAMES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERSON_H,
    DEFAULT_MAP_KSIZE,
    DEFAULT_MAP_SIGMA,
    DEFAULT_VALID_THR,
    IMG_ORI_H,
    IMG_ORI_W,
)
from dataset import create_wildtrack_dataset
from geometry import build_mvdet_proj_mat, compute_valid_ratio_from_homography, make_worldgrid2worldcoord_mat
from loss import GaussianMSE
from models import create_model
from utils import build_gaussian_kernel_2d


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MVDet 风格模型评估脚本（对齐 src 主链路）")

    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Wildtrack 数据集根目录")
    ap.add_argument("--views", type=str, default="0,1,2", help="使用的视角 ID，例如 0,1,2")
    ap.add_argument("--drop_bad_views", action="store_true", help="是否丢弃低有效性的视角")
    ap.add_argument("--valid_thr", type=float, default=DEFAULT_VALID_THR, help="投影有效性阈值")

    ap.add_argument(
        "--model_path",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "model_final.pth"),
        help="模型权重路径（支持纯 state_dict 或 checkpoint）",
    )
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")

    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES, help="评估使用的最大帧数")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="批大小")
    ap.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="数据加载线程数")

    ap.add_argument("--bev_down", type=int, default=DEFAULT_BEV_DOWN, help="BEV 下采样倍数")
    ap.add_argument("--feat_h", type=int, default=DEFAULT_FEAT_H, help="特征平面高度")
    ap.add_argument("--feat_w", type=int, default=DEFAULT_FEAT_W, help="特征平面宽度")
    ap.add_argument("--img_h", type=int, default=DEFAULT_IMG_H, help="输入图像高度")
    ap.add_argument("--img_w", type=int, default=DEFAULT_IMG_W, help="输入图像宽度")
    ap.add_argument("--person_h", type=float, default=DEFAULT_PERSON_H, help="人体高度（米）")

    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="图像辅助损失权重")
    ap.add_argument("--map_ksize", type=int, default=DEFAULT_MAP_KSIZE, help="BEV 热图高斯核大小")
    ap.add_argument("--map_sigma", type=float, default=DEFAULT_MAP_SIGMA, help="BEV 热图高斯标准差")
    ap.add_argument("--img_ksize", type=int, default=DEFAULT_IMG_KSIZE, help="图像热图高斯核大小")
    ap.add_argument("--img_sigma", type=float, default=DEFAULT_IMG_SIGMA, help="图像热图高斯标准差")
    ap.add_argument("--log_every", type=int, default=20, help="每多少步打印一次评估日志")

    return ap.parse_args()


def _build_projection(
    data_root: Path,
    views: list[int],
    feat_hw: Tuple[int, int],
    bev_down: int,
    drop_bad_views: bool,
    valid_thr: float,
) -> Tuple[Dict[int, Dict], torch.Tensor, list[int], float, Tuple[int, int], Dict[str, float]]:
    pom = parse_rectangles_pom(data_root / "rectangles.pom")

    calib_loader = CalibrationLoader(data_root / "calibrations", CAM_NAMES)
    calib_cache, t_norms = calib_loader.load_all(views)

    step_m = float(pom.get("STEP", 0.025))
    unit_scale = decide_unit_scale(step_m, t_norms)
    print(f"[UNIT] step={step_m}, median||t||={np.median(t_norms):.2f} => unit_scale={unit_scale}")

    Hb = int(pom.get("NB_HEIGHT", 1440)) // bev_down
    Wb = int(pom.get("NB_WIDTH", 480)) // bev_down
    Hf, Wf = feat_hw

    sx_f = Wf / IMG_ORI_W
    sy_f = Hf / IMG_ORI_H
    for v in views:
        calib_cache[v]["K_feat"] = scale_intrinsics(calib_cache[v]["K0"], sx=sx_f, sy=sy_f)

    origin_x_m = float(pom.get("ORIGINE_X", -3.0))
    origin_y_m = float(pom.get("ORIGINE_Y", -9.0))
    step = (step_m * bev_down) * unit_scale
    ox = origin_x_m * unit_scale
    oy = origin_y_m * unit_scale
    w2w_mat = make_worldgrid2worldcoord_mat(ox, oy, step)

    proj_mats = []
    kept_views = []
    for v in views:
        K_feat = calib_cache[v]["K_feat"]
        R = calib_cache[v]["R"]
        t = calib_cache[v]["t"]

        try:
            proj = build_mvdet_proj_mat(K_feat, R, t, w2w_mat)
        except np.linalg.LinAlgError:
            print(f"[GRID] view={v} cam={calib_cache[v]['cam']} singular")
            if drop_bad_views:
                continue
            raise RuntimeError("投影矩阵奇异")

        vr = compute_valid_ratio_from_homography(proj, (Hf, Wf), (Hb, Wb))
        print(f"[GRID] view={v} cam={calib_cache[v]['cam']} valid_ratio={vr:.4f}")
        if drop_bad_views and vr < valid_thr:
            print(f"[GRID] drop view={v}")
            continue

        proj_mats.append(torch.from_numpy(proj).float())
        kept_views.append(v)

    if not kept_views:
        raise RuntimeError("没有有效视角可用于评估")

    return calib_cache, torch.stack(proj_mats, dim=0), kept_views, unit_scale, (Hb, Wb), pom


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    map_kernel: torch.Tensor,
    img_kernel: torch.Tensor,
    alpha: float,
    log_every: int = 20,
) -> Dict[str, float]:
    criterion = GaussianMSE()
    losses, bev_losses, img_losses = [], [], []
    pos_mses, aux_pos_mses = [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, (_, x_views, map_gt, imgs_gt) in enumerate(dataloader):
            x_views = x_views.to(device, non_blocking=True)
            map_gt = map_gt.to(device, non_blocking=True)
            imgs_gt = imgs_gt.to(device, non_blocking=True)

            map_logits, imgs_logits = model(x_views)
            map_res = torch.sigmoid(map_logits)
            imgs_res = torch.sigmoid(imgs_logits)

            bev_loss = criterion(map_res, map_gt, map_kernel)
            per_view_loss = 0.0
            for vi in range(imgs_res.shape[1]):
                per_view_loss = per_view_loss + criterion(imgs_res[:, vi], imgs_gt[:, vi], img_kernel)
            per_view_loss = per_view_loss / float(imgs_res.shape[1])
            loss = bev_loss + alpha * per_view_loss

            losses.append(loss.item())
            bev_losses.append(bev_loss.item())
            img_losses.append(per_view_loss.item())

            pooled_gt = F.adaptive_max_pool2d(map_gt, output_size=map_res.shape[-2:])
            pos_mask = pooled_gt > 0.1
            pos_mse = ((map_res - pooled_gt) ** 2)[pos_mask].mean().item() if pos_mask.any() else float("nan")
            pos_mses.append(pos_mse)

            aux_pos_mask = imgs_gt > 0.1
            aux_pos_mse = ((imgs_res - imgs_gt) ** 2)[aux_pos_mask].mean().item() if aux_pos_mask.any() else float(
                "nan"
            )
            aux_pos_mses.append(aux_pos_mse)

            if batch_idx % log_every == 0:
                print(
                    f"[eval step {batch_idx}] "
                    f"loss={loss.item():.6f} "
                    f"bev={bev_loss.item():.6f} "
                    f"img={per_view_loss.item():.6f} "
                    f"pos_mse={pos_mse:.6f} "
                    f"aux_pos_mse={aux_pos_mse:.6f}"
                )

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "bev_loss": float(np.mean(bev_losses)) if bev_losses else float("nan"),
        "img_loss": float(np.mean(img_losses)) if img_losses else float("nan"),
        "pos_mse": float(np.nanmean(pos_mses)) if pos_mses else float("nan"),
        "aux_pos_mse": float(np.nanmean(aux_pos_mses)) if aux_pos_mses else float("nan"),
    }


def _load_model_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=True)


def main() -> Dict[str, float]:
    args = parse_args()
    dev = torch.device(args.device)
    print(f"[DEV] device={dev}, cuda_available={torch.cuda.is_available()}")
    if dev.type == "cuda":
        print(f"[DEV] gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    views = [int(x.strip()) for x in args.views.split(",") if x.strip().isdigit()]
    if not views:
        raise ValueError("至少需要一个视角")
    print(f"[CFG] views={views}")

    feat_hw = (args.feat_h, args.feat_w)
    calib_cache, proj_mats, kept_views, unit_scale, reduced_hw, _ = _build_projection(
        data_root=data_root,
        views=views,
        feat_hw=feat_hw,
        bev_down=args.bev_down,
        drop_bad_views=args.drop_bad_views,
        valid_thr=args.valid_thr,
    )

    print(f"[CFG] img={args.img_h}x{args.img_w}, feat={args.feat_h}x{args.feat_w}, bev={reduced_hw}")
    print(f"[CFG] kept_views={kept_views}")

    ds = create_wildtrack_dataset(
        data_root=data_root,
        views=kept_views,
        max_frames=args.max_frames,
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

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(f"[DATA] len(ds)={len(ds)}, len(loader)={len(loader)}")

    model = create_model(
        num_views=len(kept_views),
        proj_mats=proj_mats.to(dev),
        reduced_hw=reduced_hw,
        feat_hw=feat_hw,
        device=dev,
        pretrained=False,
        feat_ch=DEFAULT_FEAT_CH,
        add_coord=True,
    )
    _load_model_weights(model, model_path, dev)
    print(f"[MODEL] loaded weights from {model_path}")

    map_kernel = build_gaussian_kernel_2d(args.map_ksize, args.map_sigma, device=dev)
    img_kernel = build_gaussian_kernel_2d(args.img_ksize, args.img_sigma, device=dev)

    metrics = evaluate_model(
        model=model,
        dataloader=loader,
        device=dev,
        map_kernel=map_kernel,
        img_kernel=img_kernel,
        alpha=args.alpha,
        log_every=args.log_every,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 56)
    print("评估结果")
    print("=" * 56)
    print(f"loss:          {metrics['loss']:.6f}")
    print(f"bev_loss:      {metrics['bev_loss']:.6f}")
    print(f"img_loss:      {metrics['img_loss']:.6f}")
    print(f"pos_mse:       {metrics['pos_mse']:.6f}")
    print(f"aux_pos_mse:   {metrics['aux_pos_mse']:.6f}")
    print(f"model_size_mb: {model_size_mb:.2f}")
    print(f"total_params:  {total_params}")
    print(f"trainable:     {trainable_params}")
    print("=" * 56)

    return {
        **metrics,
        "model_size_mb": model_size_mb,
        "total_params": float(total_params),
        "trainable_params": float(trainable_params),
    }


if __name__ == "__main__":
    main()
