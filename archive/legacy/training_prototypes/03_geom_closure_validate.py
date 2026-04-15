import json
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import cv2
from PIL import Image, ImageDraw

try:
    from configs.config import DATA_ROOT, OUT_DIR
except Exception:
    BASE = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE / "wildtrack"
    OUT_DIR = BASE / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 固定的口径（你当前验证下来 scale=100 最合理）
# positionID -> (x,y) 先按你原来“xy”方式（NB_WIDTH=480, NB_HEIGHT=1440）
# 你之前 rectangles.pom 写的是 ORIGINE_X=-3.0, ORIGINE_Y=-9.0 (m)
# -------------------------
ORIG_X_M = -3.0
ORIG_Y_M = -9.0
NB_W = 480
NB_H = 1440
STEP_M = 12.0 / NB_W  # 0.025 m

SCALE = 100.0       # meters -> centimeters
INVERT_EXTR = False # 固定不反演
VARIANT = "xy"      # 固定

CAM_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]


def posid_to_xy_m(pos_id: int) -> tuple[float, float]:
    # variant="xy": ix=id%480, iy=id//480
    ix = pos_id % NB_W
    iy = pos_id // NB_W
    x = ORIG_X_M + ix * STEP_M
    y = ORIG_Y_M + iy * STEP_M
    return float(x), float(y)


def read_opencv_xml_vec(xml_path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    node = fs.getNode(key)
    if node.empty():
        fs.release()
        raise ValueError(f"Key '{key}' not found in {xml_path}")

    # extr 中 rvec/tvec 是 sequence
    if node.isSeq():
        vals = [node.at(i).real() for i in range(node.size())]
        arr = np.array(vals, dtype=np.float64)
    else:
        arr = np.array(node.mat(), dtype=np.float64).reshape(-1)

    fs.release()
    return arr


def read_intrinsics(xml_path: Path):
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    K = np.array(K, dtype=np.float64)
    dist = np.array(dist, dtype=np.float64).reshape(-1)
    return K, dist


def project_point_with_depth(K, dist, rvec, tvec, Pw_cm: np.ndarray):
    """
    Pw_cm: (3,) 世界坐标（cm），与 tvec 单位一致
    返回: (u,v,z_cam)
    """
    R, _ = cv2.Rodrigues(rvec)
    Xc = R @ Pw_cm.reshape(3, 1) + tvec
    z = float(Xc[2, 0])
    imgpts, _ = cv2.projectPoints(Pw_cm.reshape(1, 1, 3), rvec, tvec, K, dist)
    u, v = imgpts.reshape(-1)
    return float(u), float(v), z


def percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else float("nan")


def main(frame_index=0, max_people=0, margin=200, save_overlay=True):
    ann_dir = DATA_ROOT / "annotations_positions"
    img_root = DATA_ROOT / "Image_subsets"
    calib_root = DATA_ROOT / "calibrations"

    out_dir = OUT_DIR / "geom_closure_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ann_dir.glob("*.json"))
    assert files, f"No json found in {ann_dir}"
    assert 0 <= frame_index < len(files), f"frame_index out of range: {frame_index}, total={len(files)}"

    json_path = files[frame_index]
    stem = json_path.stem
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "annotations_positions 单帧 json 顶层应为 list"

    # viewNum -> list[(pid, pos_id, bbox)]
    per_view = defaultdict(list)
    cnt = 0
    for obj in data:
        pid = obj.get("personID", None)
        pos_id = obj.get("positionID", None)
        views = obj.get("views", [])
        if pid is None or pos_id is None or not isinstance(views, list):
            continue

        for v in views:
            if not isinstance(v, dict) or "viewNum" not in v:
                continue
            if not all(k in v for k in ["xmin", "ymin", "xmax", "ymax"]):
                continue
            vn = int(v["viewNum"])
            bbox = (int(v["xmin"]), int(v["ymin"]), int(v["xmax"]), int(v["ymax"]))
            per_view[vn].append((int(pid), int(pos_id), bbox))

        cnt += 1
        if max_people > 0 and cnt >= max_people:
            break

    assert per_view, "该帧没有读到任何 view 标注，检查 json 解析逻辑"

    # 读取相机参数
    cam_params = {}
    for i, name in enumerate(CAM_NAMES):
        intr = calib_root / "intrinsic_original" / f"intr_{name}.xml"
        extr = calib_root / "extrinsic" / f"extr_{name}.xml"
        assert intr.exists(), f"Missing intrinsic: {intr}"
        assert extr.exists(), f"Missing extrinsic: {extr}"

        K, dist = read_intrinsics(intr)
        rvec = read_opencv_xml_vec(extr, "rvec").reshape(3, 1)
        tvec = read_opencv_xml_vec(extr, "tvec").reshape(3, 1)
        cam_params[i] = (name, K, dist, rvec, tvec)

    # 统计：像素脚点误差（投影点 vs bbox底边中心）
    all_err = []
    all_err_by_cam = defaultdict(list)
    total_all = 0
    valid_all = 0

    overlays = {}  # vn -> (img, draw)

    for vn, items in per_view.items():
        name, K, dist, rvec, tvec = cam_params[vn]

        # 打开图（用于 overlay）
        if save_overlay:
            cam_folder = f"C{vn + 1}"
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                p = img_root / cam_folder / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is not None:
                img = Image.open(img_path).convert("RGB")
                overlays[vn] = (img, ImageDraw.Draw(img))
                W, H = img.size
            else:
                W, H = None, None
        else:
            W, H = None, None

        for pid, pos_id, bbox in items:
            total_all += 1

            xmin, ymin, xmax, ymax = bbox
            # 观测脚点：bbox底边中心
            u_obs = 0.5 * (xmin + xmax)
            v_obs = float(ymax)

            # 预测脚点：positionID -> 世界点 -> 投影
            x_m, y_m = posid_to_xy_m(pos_id)
            Pw_cm = np.array([x_m * SCALE, y_m * SCALE, 0.0], dtype=np.float64)

            u_pred, v_pred, z_cam = project_point_with_depth(K, dist, rvec, tvec, Pw_cm)

            # 基本有效性过滤：深度必须为正
            if z_cam <= 1e-6:
                continue

            # 如果有图像尺寸，做边界过滤（给 margin 缓冲）
            if W is not None and H is not None:
                if (u_pred < -margin) or (u_pred > W + margin) or (v_pred < -margin) or (v_pred > H + margin):
                    continue

            err = float(np.hypot(u_pred - u_obs, v_pred - v_obs))
            all_err.append(err)
            all_err_by_cam[vn].append(err)
            valid_all += 1

            if save_overlay and vn in overlays:
                img, draw = overlays[vn]
                # bbox（白框）
                draw.rectangle([xmin, ymin, xmax, ymax], width=2)
                # 观测脚点（绿点）
                r0 = 3
                draw.ellipse([u_obs - r0, v_obs - r0, u_obs + r0, v_obs + r0], outline=(0, 255, 0), width=2)
                # 预测脚点（红点）
                r = 4
                draw.ellipse([u_pred - r, v_pred - r, u_pred + r, v_pred + r], outline=(255, 0, 0), width=2)

    all_err = np.array(all_err, dtype=np.float64)
    print(f"[CONFIG] scale={SCALE} (m->cm), inv={INVERT_EXTR}, variant={VARIANT}")
    print(f"[VALID] {valid_all}/{total_all} used after filtering (depth>0 and in-range)")
    if all_err.size:
        print(f"[ERR px] median={np.median(all_err):.2f}  p90={percentile(all_err,90):.2f}  p95={percentile(all_err,95):.2f}  mean={np.mean(all_err):.2f}")
    else:
        print("[ERR px] No valid samples. Something is wrong (projection out of range or depth negative).")

    # 每个相机也打印一下（便于发现某个相机标定/映射异常）
    for vn in range(7):
        arr = np.array(all_err_by_cam.get(vn, []), dtype=np.float64)
        if arr.size:
            print(f"  - cam C{vn+1} {CAM_NAMES[vn]}: n={arr.size} median={np.median(arr):.2f} p90={percentile(arr,90):.2f}")
        else:
            print(f"  - cam C{vn+1} {CAM_NAMES[vn]}: n=0")

    # 保存 overlay 图
    if save_overlay:
        for vn, (img, _) in overlays.items():
            name = CAM_NAMES[vn]
            out_path = out_dir / f"{stem}_C{vn+1}_{name}_scale{int(SCALE)}_footerr.png"
            img.save(out_path)
            print("[OK] saved", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--max_people", type=int, default=0)
    parser.add_argument("--margin", type=int, default=200)
    parser.add_argument("--no_overlay", action="store_true")
    args = parser.parse_args()

    main(
        frame_index=args.frame_index,
        max_people=args.max_people,
        margin=args.margin,
        save_overlay=(not args.no_overlay),
    )
