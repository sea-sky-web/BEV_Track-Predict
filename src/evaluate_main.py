"""
Offline evaluation entrypoint for MVDet-like training chain.

This script now supports:
1) Loss-style metrics aligned with training loss.
2) Detection-style metrics (precision/recall/F1) by threshold sweep.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom, scale_intrinsics
from config import (
    CAM_NAMES,
    DEFAULT_ALPHA,
    DEFAULT_LOSS_TYPE,
    DEFAULT_FOCAL_ALPHA,
    DEFAULT_FOCAL_BETA,
    DEFAULT_BACKBONE,
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
    DEFAULT_MAP_KSIZE,
    DEFAULT_MAP_SIGMA,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MODA_DIST_M,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERSON_H,
    DEFAULT_FUSION_MODE,
    DEFAULT_VIEWS,
    DEFAULT_VALID_THR,
    IMG_ORI_H,
    IMG_ORI_W,
)
from dataset import create_wildtrack_dataset
from geometry import build_mvdet_proj_mat, compute_valid_ratio_from_homography, make_worldgrid2worldcoord_mat
from loss import GaussianMSE, create_loss_criterion
from metrics import aggregate_metrics, compute_moda_modp
from models import create_model
from utils import build_gaussian_kernel_2d


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate MVDet-like model (loss metrics + optional detection metrics)."
    )

    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Wildtrack dataset root")
    ap.add_argument("--views", type=str, default=DEFAULT_VIEWS, help="View IDs, e.g. 0,1,2")
    ap.add_argument("--drop_bad_views", action="store_true", help="Drop views with low valid projection ratio")
    ap.add_argument("--valid_thr", type=float, default=DEFAULT_VALID_THR, help="Projection valid ratio threshold")

    ap.add_argument(
        "--model_path",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "model_final.pth"),
        help="Path to model weights (.pth)",
    )
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")

    ap.add_argument("--frame_start", type=int, default=0, help="Start frame index (annotation list index)")
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES, help="Number of frames to evaluate")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Dataloader workers")

    ap.add_argument("--bev_down", type=int, default=DEFAULT_BEV_DOWN, help="BEV downsample factor")
    ap.add_argument("--feat_h", type=int, default=DEFAULT_FEAT_H, help="Feature map height")
    ap.add_argument("--feat_w", type=int, default=DEFAULT_FEAT_W, help="Feature map width")
    ap.add_argument("--img_h", type=int, default=DEFAULT_IMG_H, help="Input image height")
    ap.add_argument("--img_w", type=int, default=DEFAULT_IMG_W, help="Input image width")
    ap.add_argument("--person_h", type=float, default=DEFAULT_PERSON_H, help="Person height (meters)")
    ap.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE,
        choices=["resnet18", "resnet50"],
        help="Backbone topology used by the checkpoint.",
    )
    ap.add_argument(
        "--fusion_mode",
        type=str,
        default=os.environ.get("FUSION_MODE", DEFAULT_FUSION_MODE),
        choices=["concat", "confidence", "confidence_v1", "confidence_v2", "geo_confidence_v1"],
        help="BEV fusion mode used by the checkpoint.",
    )

    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Aux image loss weight")
    ap.add_argument("--loss_type", type=str, default=DEFAULT_LOSS_TYPE, choices=["mse", "focal"],
                    help="BEV loss type used by the checkpoint (for reference loss metrics only).")
    ap.add_argument("--focal_alpha", type=float, default=DEFAULT_FOCAL_ALPHA)
    ap.add_argument("--focal_beta", type=float, default=DEFAULT_FOCAL_BETA)
    ap.add_argument("--map_ksize", type=int, default=DEFAULT_MAP_KSIZE, help="Gaussian kernel size for BEV")
    ap.add_argument("--map_sigma", type=float, default=DEFAULT_MAP_SIGMA, help="Gaussian sigma for BEV")
    ap.add_argument("--img_ksize", type=int, default=DEFAULT_IMG_KSIZE, help="Gaussian kernel size for image")
    ap.add_argument("--img_sigma", type=float, default=DEFAULT_IMG_SIGMA, help="Gaussian sigma for image")
    ap.add_argument("--log_every", type=int, default=20, help="Log interval")

    ap.add_argument("--report_detection", action="store_true", help="Report detection metrics")
    ap.add_argument(
        "--det_thresholds",
        type=str,
        default="-0.50,-0.25,-0.10,0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50",
        help="Comma-separated detection thresholds (raw logit space, MVDet default=0.4)",
    )
    ap.add_argument("--det_dist_thr", type=float, default=3.0, help="Match distance threshold (BEV cells)")
    ap.add_argument("--det_moda_dist_m", type=float, default=DEFAULT_MODA_DIST_M, help="MODA/MODP match threshold in meters")
    ap.add_argument("--det_nms_ksize", type=int, default=3, help="NMS kernel size (odd int, used only for GT extraction)")
    ap.add_argument(
        "--det_min_distance",
        type=float,
        default=6.0,
        help="Greedy NMS suppression radius in reduced BEV cells. "
             "Grid-sweep optimal: 6.0 (MODA=0.857). "
             "MVDet equivalent: 5.0 (50cm / 10cm-per-reduced-cell).",
    )
    ap.add_argument(
        "--det_min_distances",
        type=str,
        default="",
        help="Comma-separated NMS radii to sweep (e.g. 3.0,4.0,5.0,6.0,7.0,8.0). "
             "When set, overrides --det_min_distance and runs a full threshold sweep per radius.",
    )
    ap.add_argument("--det_max_preds", type=int, default=0, help="Max predictions per frame (0=unlimited, MVDet uses top_k sort then NMS)")
    ap.add_argument(
        "--use_offset",
        action="store_true",
        help="Apply the model's offset head prediction to refine detection point coordinates.",
    )

    ap.add_argument(
        "--metrics_out",
        type=str,
        default="",
        help="Optional path to save metrics json",
    )

    return ap.parse_args()


def parse_views(raw: str) -> List[int]:
    views = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    if not views:
        raise ValueError("At least one valid view id is required.")
    return views


def parse_thresholds(raw: str) -> List[float]:
    vals: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("det_thresholds cannot be empty.")
    vals = sorted(set(vals))
    return vals


def parse_min_distances(raw: str) -> List[float]:
    vals: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    return sorted(set(vals))


def _build_projection(
    data_root: Path,
    views: Sequence[int],
    feat_hw: Tuple[int, int],
    bev_down: int,
    drop_bad_views: bool,
    valid_thr: float,
) -> Tuple[Dict[int, Dict], torch.Tensor, List[int], float, Tuple[int, int], Dict, float]:
    pom = parse_rectangles_pom(data_root / "rectangles.pom")

    calib_loader = CalibrationLoader(data_root / "calibrations", CAM_NAMES)
    calib_cache, t_norms = calib_loader.load_all(list(views))

    step_m = float(pom.get("STEP", 0.025))
    unit_scale = decide_unit_scale(step_m, t_norms)
    print(f"[UNIT] step={step_m}, median||t||={np.median(t_norms):.2f} => unit_scale={unit_scale}")

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

    proj_mats: List[torch.Tensor] = []
    kept_views: List[int] = []
    for v in views:
        k_feat = calib_cache[v]["K_feat"]
        r = calib_cache[v]["R"]
        t = calib_cache[v]["t"]

        try:
            proj = build_mvdet_proj_mat(k_feat, r, t, w2w_mat)
        except np.linalg.LinAlgError:
            print(f"[GRID] view={v} cam={calib_cache[v]['cam']} singular")
            if drop_bad_views:
                continue
            raise RuntimeError("Projection matrix is singular.")

        vr = compute_valid_ratio_from_homography(proj, (hf, wf), (hb, wb))
        print(f"[GRID] view={v} cam={calib_cache[v]['cam']} valid_ratio={vr:.4f}")
        if drop_bad_views and vr < valid_thr:
            print(f"[GRID] drop view={v}")
            continue

        proj_mats.append(torch.from_numpy(proj).float())
        kept_views.append(v)

    if not kept_views:
        raise RuntimeError("No valid view remained after projection filtering.")

    return calib_cache, torch.stack(proj_mats, dim=0), kept_views, unit_scale, (hb, wb), pom, step_m


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    map_kernel: torch.Tensor,
    img_kernel: torch.Tensor,
    alpha: float,
    log_every: int = 20,
    collect_detection_inputs: bool = False,
    loss_type: str = "mse",
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
) -> Tuple[Dict[str, float], List[torch.Tensor], List[torch.Tensor]]:
    criterion = create_loss_criterion(loss_type=loss_type, focal_alpha=focal_alpha, focal_beta=focal_beta)
    losses: List[float] = []
    bev_losses: List[float] = []
    img_losses: List[float] = []
    pos_mses: List[float] = []
    aux_pos_mses: List[float] = []

    pred_maps: List[torch.Tensor] = []
    gt_maps: List[torch.Tensor] = []
    offset_maps: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (_, x_views, map_gt, imgs_gt) in enumerate(dataloader):
            x_views = x_views.to(device, non_blocking=True)
            map_gt = map_gt.to(device, non_blocking=True)
            imgs_gt = imgs_gt.to(device, non_blocking=True)

            map_logits, offset_preds, imgs_logits = model(x_views)

            # Loss: use raw logits (matches training, matches MVDet)
            bev_loss = criterion(map_logits, map_gt, map_kernel)
            per_view_loss = 0.0
            for vi in range(imgs_logits.shape[1]):
                per_view_loss = per_view_loss + criterion(imgs_logits[:, vi], imgs_gt[:, vi], img_kernel)
            per_view_loss = per_view_loss / float(imgs_logits.shape[1])
            loss = bev_loss + alpha * per_view_loss

            losses.append(loss.item())
            bev_losses.append(bev_loss.item())
            img_losses.append(per_view_loss.item())

            pooled_gt = F.adaptive_max_pool2d(map_gt, output_size=map_logits.shape[-2:])
            pos_mask = pooled_gt > 0.1
            pos_mse = ((map_logits - pooled_gt) ** 2)[pos_mask].mean().item() if pos_mask.any() else float("nan")
            pos_mses.append(pos_mse)

            aux_pos_mask = imgs_gt > 0.1
            aux_pos_mse = ((imgs_logits - imgs_gt) ** 2)[aux_pos_mask].mean().item() if aux_pos_mask.any() else float(
                "nan"
            )
            aux_pos_mses.append(aux_pos_mse)

            # Detection: use raw logits, not sigmoid (MVDet convention)
            if collect_detection_inputs:
                for bi in range(map_logits.shape[0]):
                    pred_maps.append(map_logits[bi, 0].detach().cpu())
                    gt_maps.append(pooled_gt[bi, 0].detach().cpu())
                    offset_maps.append(offset_preds[bi].detach().cpu())

            if batch_idx % log_every == 0:
                print(
                    f"[eval step {batch_idx}] "
                    f"loss={loss.item():.6f} "
                    f"bev={bev_loss.item():.6f} "
                    f"img={per_view_loss.item():.6f} "
                    f"pos_mse={pos_mse:.6f} "
                    f"aux_pos_mse={aux_pos_mse:.6f}"
                )

    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "bev_loss": float(np.mean(bev_losses)) if bev_losses else float("nan"),
        "img_loss": float(np.mean(img_losses)) if img_losses else float("nan"),
        "pos_mse": float(np.nanmean(pos_mses)) if pos_mses else float("nan"),
        "aux_pos_mse": float(np.nanmean(aux_pos_mses)) if aux_pos_mses else float("nan"),
    }
    return metrics, pred_maps, gt_maps, offset_maps


def _extract_points(
    heatmap: torch.Tensor,
    threshold: float,
    nms_ksize: int,
    max_preds: int,
    min_distance: float = 5.0,
    offset_map: torch.Tensor | None = None,
) -> np.ndarray:
    """Extract detection points from a BEV heatmap using MVDet-style greedy distance NMS.

    MVDet reference: github.com/hou-yz/MVDet/blob/master/multiview_detector/utils/nms.py
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {tuple(heatmap.shape)}")

    hm = heatmap.float()
    ys, xs = torch.nonzero(hm >= threshold, as_tuple=True)
    if ys.numel() == 0:
        return np.empty((0, 3), dtype=np.float32)

    scores = hm[ys, xs]
    order = torch.argsort(scores, descending=True)
    ys = ys[order].float()
    xs = xs[order].float()
    scores = scores[order].float()

    if max_preds > 0:
        ys = ys[:max_preds]
        xs = xs[:max_preds]
        scores = scores[:max_preds]

    # Greedy distance-based NMS (MVDet style)
    if min_distance > 0.0 and ys.numel() > 1:
        keep_mask = torch.ones(ys.numel(), dtype=torch.bool, device=ys.device)
        for i in range(ys.numel()):
            if not keep_mask[i]:
                continue
            dy = ys[i + 1:] - ys[i]
            dx = xs[i + 1:] - xs[i]
            too_close = (dy * dy + dx * dx) < (min_distance * min_distance)
            if too_close.any():
                suppress_idx = torch.nonzero(too_close, as_tuple=False).view(-1) + i + 1
                keep_mask[suppress_idx] = False
        ys = ys[keep_mask]
        xs = xs[keep_mask]
        scores = scores[keep_mask]

    ys_np = ys.cpu().numpy().astype(np.float32)
    xs_np = xs.cpu().numpy().astype(np.float32)
    scores_np = scores.cpu().numpy().astype(np.float32)

    if offset_map is not None and ys_np.size > 0:
        ys_int = ys.long().cpu()
        xs_int = xs.long().cpu()
        ys_np = ys_np + offset_map[0, ys_int, xs_int].numpy()
        xs_np = xs_np + offset_map[1, ys_int, xs_int].numpy()

    return np.stack([ys_np, xs_np, scores_np], axis=1)


def _extract_gt_points(heatmap: torch.Tensor, nms_ksize: int) -> np.ndarray:
    pts = _extract_points(
        heatmap.float(),
        threshold=0.5,
        nms_ksize=nms_ksize,
        max_preds=0,
        min_distance=0.0,
    )
    return pts[:, :2] if pts.size else np.empty((0, 2), dtype=np.float32)


def _match_points(pred_yx: np.ndarray, gt_yx: np.ndarray, dist_thr: float) -> Tuple[int, int, int, float]:
    if pred_yx.shape[0] == 0:
        return 0, 0, int(gt_yx.shape[0]), 0.0
    if gt_yx.shape[0] == 0:
        return 0, int(pred_yx.shape[0]), 0, 0.0

    used = np.zeros(gt_yx.shape[0], dtype=bool)
    tp = 0
    dist_sum = 0.0

    for pi in range(pred_yx.shape[0]):
        dists = np.sqrt(((gt_yx - pred_yx[pi]) ** 2).sum(axis=1))
        dists[used] = np.inf
        gi = int(np.argmin(dists))
        if np.isfinite(dists[gi]) and float(dists[gi]) <= dist_thr:
            used[gi] = True
            tp += 1
            dist_sum += float(dists[gi])

    fp = int(pred_yx.shape[0] - tp)
    fn = int(gt_yx.shape[0] - tp)
    return tp, fp, fn, dist_sum


def evaluate_detection(
    pred_maps: Sequence[torch.Tensor],
    gt_maps: Sequence[torch.Tensor],
    thresholds: Sequence[float],
    dist_thr: float,
    nms_ksize: int,
    max_preds: int,
    bev_cell_m: float,
    min_distance: float,
    moda_dist_m: float,
    offset_maps: Sequence[torch.Tensor] | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if len(pred_maps) != len(gt_maps):
        raise ValueError("pred_maps and gt_maps must have the same length.")
    if offset_maps is not None and len(offset_maps) != len(pred_maps):
        raise ValueError("offset_maps and pred_maps must have the same length.")

    for thr in thresholds:
        tp = 0
        fp = 0
        fn = 0
        dist_sum = 0.0
        moda_rows: List[Dict[str, float]] = []

        for idx, (pred_map, gt_map) in enumerate(zip(pred_maps, gt_maps)):
            pred_pts = _extract_points(
                pred_map,
                thr,
                nms_ksize=nms_ksize,
                max_preds=max_preds,
                min_distance=min_distance,
                offset_map=offset_maps[idx] if offset_maps is not None else None,
            )
            pred_yx = pred_pts[:, :2] if pred_pts.size else np.empty((0, 2), dtype=np.float32)

            gt_yx = _extract_gt_points(gt_map, nms_ksize=nms_ksize)

            c_tp, c_fp, c_fn, c_dist = _match_points(pred_yx=pred_yx, gt_yx=gt_yx, dist_thr=dist_thr)
            tp += c_tp
            fp += c_fp
            fn += c_fn
            dist_sum += c_dist
            moda_rows.append(
                compute_moda_modp(
                    pred_pts=pred_yx * bev_cell_m,
                    gt_pts=gt_yx * bev_cell_m,
                    d_thresh=moda_dist_m,
                )
            )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        loc_err_px = dist_sum / tp if tp > 0 else float("nan")
        loc_err_m = loc_err_px * bev_cell_m if np.isfinite(loc_err_px) else float("nan")
        moda_metrics = aggregate_metrics(moda_rows)

        rows.append(
            {
                "threshold": float(thr),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "loc_err_px": float(loc_err_px),
                "loc_err_m": float(loc_err_m),
                "moda": float(moda_metrics["moda"]),
                "modp": float(moda_metrics["modp"]),
                "moda_tp": float(moda_metrics["tp"]),
                "moda_fp": float(moda_metrics["fp"]),
                "moda_fn": float(moda_metrics["fn"]),
                "moda_n_gt": float(moda_metrics["n_gt"]),
            }
        )

    best = max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"]))
    return rows, best


def _print_detection_table(rows: Sequence[Dict[str, float]], best_threshold: float) -> None:
    print("\n" + "=" * 56)
    print("Detection Sweep (BEV)")
    print("=" * 56)
    print("thr      precision  recall     f1         moda       modp       loc_err(m)")
    for row in rows:
        mark = "*" if abs(row["threshold"] - best_threshold) < 1e-12 else " "
        loc_m = "nan" if not np.isfinite(row["loc_err_m"]) else f"{row['loc_err_m']:.4f}"
        modp = "nan" if not np.isfinite(row["modp"]) else f"{row['modp']:.4f}"
        print(
            f"{mark}{row['threshold']:<8.2f} "
            f"{row['precision']:<10.4f} "
            f"{row['recall']:<10.4f} "
            f"{row['f1']:<10.4f} "
            f"{row['moda']:<10.4f} "
            f"{modp:<10} "
            f"{loc_m}"
        )
    print("=" * 56)


def _load_model_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint shape mismatch: training and evaluation model topology differ.\n"
            "Most common cause: different kept views between train/eval.\n"
            "Use exactly the same --views/--drop_bad_views/--valid_thr as training.\n"
            f"Original error:\n{exc}"
        ) from exc

    non_offset_missing = [k for k in missing_keys if not k.startswith("offset_head.")]
    if non_offset_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint shape mismatch: training and evaluation model topology differ.\n"
            "Most common cause: different kept views between train/eval, or a real "
            "architecture change (not just the offset_head addition).\n"
            "Use exactly the same --views/--drop_bad_views/--valid_thr as training.\n"
            f"Missing keys: {non_offset_missing}\n"
            f"Unexpected keys: {list(unexpected_keys)}"
        )

    if missing_keys:
        print(
            f"[MODEL] WARNING: checkpoint predates offset_head "
            f"({len(missing_keys)} keys randomly initialized: {missing_keys}). "
            "Offset refinement will be inaccurate for this checkpoint; "
            "omit --use_offset or retrain to get calibrated offsets."
        )


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
        raise FileNotFoundError(f"Model file not found: {model_path}")

    views = parse_views(args.views)
    print(f"[CFG] views={views}")

    feat_hw = (args.feat_h, args.feat_w)
    calib_cache, proj_mats, kept_views, unit_scale, reduced_hw, _, step_m = _build_projection(
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
        frame_start=args.frame_start,
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
        backbone=args.backbone,
        feat_ch=DEFAULT_FEAT_CH,
        add_coord=True,
        fusion_mode=args.fusion_mode,
    )
    _load_model_weights(model, model_path, dev)
    print(f"[MODEL] loaded weights from {model_path} backbone={args.backbone} fusion_mode={model.fusion_mode}")

    map_kernel = build_gaussian_kernel_2d(args.map_ksize, args.map_sigma, device=dev)
    img_kernel = build_gaussian_kernel_2d(args.img_ksize, args.img_sigma, device=dev)

    base_metrics, pred_maps, gt_maps, offset_maps = evaluate_model(
        model=model,
        dataloader=loader,
        device=dev,
        map_kernel=map_kernel,
        img_kernel=img_kernel,
        alpha=args.alpha,
        log_every=args.log_every,
        collect_detection_inputs=args.report_detection,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_beta=args.focal_beta,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 56)
    print("Evaluation Summary")
    print("=" * 56)
    print(f"loss:          {base_metrics['loss']:.6f}")
    print(f"bev_loss:      {base_metrics['bev_loss']:.6f}")
    print(f"img_loss:      {base_metrics['img_loss']:.6f}")
    print(f"pos_mse:       {base_metrics['pos_mse']:.6f}")
    print(f"aux_pos_mse:   {base_metrics['aux_pos_mse']:.6f}")
    print(f"model_size_mb: {model_size_mb:.2f}")
    print(f"total_params:  {total_params}")
    print(f"trainable:     {trainable_params}")
    print("=" * 56)

    final_metrics: Dict[str, float] = {
        **base_metrics,
        "model_size_mb": model_size_mb,
        "total_params": float(total_params),
        "trainable_params": float(trainable_params),
    }
    extraction_config: Dict[str, object] = {
        "det_min_distance": float(args.det_min_distance),
        "det_nms_ksize": int(args.det_nms_ksize),
        "det_max_preds": int(args.det_max_preds),
        "det_dist_thr": float(args.det_dist_thr),
        "det_moda_dist_m": float(args.det_moda_dist_m),
        "det_thresholds": args.det_thresholds,
    }
    output_payload: Dict[str, object] = {**final_metrics, **extraction_config, "extraction_config": extraction_config}

    if args.report_detection:
        thresholds = parse_thresholds(args.det_thresholds)
        bev_cell_m = step_m * args.bev_down

        nms_radii = parse_min_distances(args.det_min_distances) if args.det_min_distances else [args.det_min_distance]

        global_best = None
        global_best_nms = None
        all_sweep_results = []

        for nms_r in nms_radii:
            rows, best = evaluate_detection(
                pred_maps=pred_maps,
                gt_maps=gt_maps,
                thresholds=thresholds,
                dist_thr=args.det_dist_thr,
                nms_ksize=args.det_nms_ksize,
                max_preds=args.det_max_preds,
                bev_cell_m=bev_cell_m,
                min_distance=nms_r,
                moda_dist_m=args.det_moda_dist_m,
                offset_maps=offset_maps if args.use_offset else None,
            )

            if len(nms_radii) > 1:
                print(f"\n[NMS={nms_r:.1f}] best: thr={best['threshold']:.3f}, "
                      f"moda={best['moda']:.4f}, f1={best['f1']:.4f}, "
                      f"prec={best['precision']:.4f}, recall={best['recall']:.4f}, "
                      f"tp={int(best['moda_tp'])}, fp={int(best['moda_fp'])}, fn={int(best['moda_fn'])}")

            for row in rows:
                row["nms_radius"] = float(nms_r)
            all_sweep_results.extend(rows)

            if global_best is None or best["moda"] > global_best["moda"]:
                global_best = best
                global_best_nms = nms_r
                global_best_rows = rows

        best = global_best
        best["nms_radius"] = float(global_best_nms)

        _print_detection_table(global_best_rows, best_threshold=best["threshold"])

        print(
            "[BEST] "
            f"nms={global_best_nms:.1f}, "
            f"thr={best['threshold']:.3f}, "
            f"precision={best['precision']:.4f}, "
            f"recall={best['recall']:.4f}, "
            f"f1={best['f1']:.4f}, "
            f"moda={best['moda']:.4f}, "
            f"modp={best['modp']:.4f}, "
            f"loc_err_m={best['loc_err_m']:.4f}"
        )

        final_metrics.update(
            {
                "det_best_threshold": float(best["threshold"]),
                "det_best_nms_radius": float(global_best_nms),
                "det_precision": float(best["precision"]),
                "det_recall": float(best["recall"]),
                "det_f1": float(best["f1"]),
                "det_moda": float(best["moda"]),
                "det_modp": float(best["modp"]),
                "det_moda_dist_m": float(args.det_moda_dist_m),
                "det_loc_err_m": float(best["loc_err_m"]),
                "det_tp": float(best["tp"]),
                "det_fp": float(best["fp"]),
                "det_fn": float(best["fn"]),
                "det_moda_tp": float(best["moda_tp"]),
                "det_moda_fp": float(best["moda_fp"]),
                "det_moda_fn": float(best["moda_fn"]),
                "det_moda_n_gt": float(best["moda_n_gt"]),
            }
        )
        output_payload.update(final_metrics)
        output_payload["det_sweep"] = all_sweep_results
        output_payload["det_best"] = best

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        print(f"[SAVE] metrics saved to {out_path}")

    return final_metrics


if __name__ == "__main__":
    main()
