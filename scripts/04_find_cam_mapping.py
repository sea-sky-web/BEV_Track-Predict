import json
from pathlib import Path
from collections import defaultdict
import argparse
import itertools

import numpy as np
import cv2

try:
    from configs.config import DATA_ROOT
except Exception:
    BASE = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE / "wildtrack"

# ===== 固定你已验证的口径 =====
ORIG_X_M = -3.0
ORIG_Y_M = -9.0
NB_W = 480
STEP_M = 12.0 / NB_W  # 0.025m
SCALE = 100.0         # m -> cm

# 标定文件的“名字池”（我们要找它们与 viewNum 的排列关系）
CAM_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

def posid_to_xy_m(pos_id: int):
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

def project_uvz(K, dist, rvec, tvec, Pw_cm: np.ndarray):
    R, _ = cv2.Rodrigues(rvec)
    Xc = R @ Pw_cm.reshape(3, 1) + tvec
    z = float(Xc[2, 0])
    imgpts, _ = cv2.projectPoints(Pw_cm.reshape(1, 1, 3), rvec, tvec, K, dist)
    u, v = imgpts.reshape(-1)
    return float(u), float(v), z

def foot_obs_from_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    u_obs = 0.5 * (xmin + xmax)
    v_obs = float(ymax)
    return float(u_obs), float(v_obs)

def robust_errors_for_frame(json_path: Path, calib_by_name: dict, margin=200):
    """
    返回: errs[viewNum] = list[err_px]
    这里只做与图像尺寸无关的过滤（无法知道W/H），仅做深度过滤 + 非常宽松的像素范围过滤
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    errs = defaultdict(list)

    for obj in data:
        pos_id = obj.get("positionID", None)
        views = obj.get("views", [])
        if pos_id is None or not isinstance(views, list):
            continue

        x_m, y_m = posid_to_xy_m(int(pos_id))
        Pw_cm = np.array([x_m * SCALE, y_m * SCALE, 0.0], dtype=np.float64)

        for v in views:
            if not isinstance(v, dict) or "viewNum" not in v:
                continue
            if not all(k in v for k in ["xmin", "ymin", "xmax", "ymax"]):
                continue

            vn = int(v["viewNum"])
            bbox = (int(v["xmin"]), int(v["ymin"]), int(v["xmax"]), int(v["ymax"]))
            u_obs, v_obs = foot_obs_from_bbox(bbox)

            # 注意：此处先不做 viewNum->name 映射；在外层用 permutation 决定
            errs[vn].append((Pw_cm, u_obs, v_obs))
    return errs

def score_mapping(per_frame_data, mapping, calib_by_name, margin=200):
    """
    mapping: tuple of 7 names, mapping[vn] = name
    目标：最小化 sum over views of median(err_px)
    返回: (total_score, per_view_medians, total_valid)
    """
    per_view_errs = {vn: [] for vn in range(7)}
    total_valid = 0

    for frame_items in per_frame_data:
        # frame_items[vn] = list of (Pw_cm, u_obs, v_obs)
        for vn in range(7):
            name = mapping[vn]
            K, dist, rvec, tvec = calib_by_name[name]
            for Pw_cm, u_obs, v_obs in frame_items.get(vn, []):
                u, v, z = project_uvz(K, dist, rvec, tvec, Pw_cm)
                if z <= 1e-6:
                    continue
                # 仅做非常宽松的像素范围过滤，避免离谱投影污染统计
                if (u < -margin) or (u > 4000 + margin) or (v < -margin) or (v > 3000 + margin):
                    continue
                err = float(np.hypot(u - u_obs, v - v_obs))
                per_view_errs[vn].append(err)
                total_valid += 1

    per_view_median = []
    total_score = 0.0
    for vn in range(7):
        arr = np.array(per_view_errs[vn], dtype=np.float64)
        if arr.size == 0:
            med = 1e9  # 没数据则惩罚
        else:
            med = float(np.median(arr))
        per_view_median.append(med)
        total_score += med

    return total_score, per_view_median, total_valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=10, help="用于搜索映射的帧数（建议10~30）")
    parser.add_argument("--start", type=int, default=0, help="从第几帧开始")
    args = parser.parse_args()

    ann_dir = DATA_ROOT / "annotations_positions"
    calib_root = DATA_ROOT / "calibrations"
    files = sorted(ann_dir.glob("*.json"))
    assert files, f"No json found in {ann_dir}"
    files = files[args.start: args.start + args.n_frames]
    assert files, "No frames selected"

    # 读入所有标定
    calib_by_name = {}
    for name in CAM_NAMES:
        intr = calib_root / "intrinsic_original" / f"intr_{name}.xml"
        extr = calib_root / "extrinsic" / f"extr_{name}.xml"
        assert intr.exists() and extr.exists(), f"Missing calib for {name}"
        K, dist = read_intrinsics(intr)
        rvec = read_opencv_xml_vec(extr, "rvec").reshape(3, 1)
        tvec = read_opencv_xml_vec(extr, "tvec").reshape(3, 1)
        calib_by_name[name] = (K, dist, rvec, tvec)

    # 预解析每帧，减少重复 IO
    per_frame_data = []
    for p in files:
        frame_items = defaultdict(list)
        data = json.loads(p.read_text(encoding="utf-8"))
        for obj in data:
            pos_id = obj.get("positionID", None)
            views = obj.get("views", [])
            if pos_id is None or not isinstance(views, list):
                continue
            x_m, y_m = posid_to_xy_m(int(pos_id))
            Pw_cm = np.array([x_m * SCALE, y_m * SCALE, 0.0], dtype=np.float64)

            for v in views:
                if not isinstance(v, dict) or "viewNum" not in v:
                    continue
                if not all(k in v for k in ["xmin", "ymin", "xmax", "ymax"]):
                    continue
                vn = int(v["viewNum"])
                bbox = (int(v["xmin"]), int(v["ymin"]), int(v["xmax"]), int(v["ymax"]))
                u_obs, v_obs = foot_obs_from_bbox(bbox)
                frame_items[vn].append((Pw_cm, u_obs, v_obs))
        per_frame_data.append(frame_items)

    best = None  # (score, mapping, per_view_medians, valid)
    perm_count = 0

    for perm in itertools.permutations(CAM_NAMES, 7):
        perm_count += 1
        score, per_view_medians, valid = score_mapping(per_frame_data, perm, calib_by_name)
        if best is None or score < best[0]:
            best = (score, perm, per_view_medians, valid)

    score, perm, per_view_medians, valid = best
    print(f"[DONE] searched {perm_count} permutations on {len(files)} frames, valid_samples={valid}")
    print(f"[BEST] total_score(sum medians)={score:.2f}")
    for vn in range(7):
        print(f"  viewNum {vn} -> {perm[vn]:7}   median_err_px={per_view_medians[vn]:.2f}")

if __name__ == "__main__":
    main()
