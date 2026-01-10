import argparse
from pathlib import Path
import numpy as np
import cv2

try:
    from configs.config import DATA_ROOT, OUT_DIR
except Exception:
    BASE = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE / "wildtrack"
    OUT_DIR = BASE / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== WILDTRACK-toolkit 明确写死的地面网格定义 =====
# grid: 1440 x 480, origin (-300, -90, 0) cm, step 2.5 cm, z=0
# 这与 rectangles.pom (ORIGINE_X=-3m, ORIGINE_Y=-9m, step=0.025m) 等价
ORIG_X_CM = -300.0
ORIG_Y_CM = -900.0
STEP_CM = 2.5
BEV_W = 480
BEV_H = 1440
Z_CM = 0.0

CAM_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

def read_intrinsics(xml_path: Path):
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    K = np.array(K, dtype=np.float64)
    dist = np.array(dist, dtype=np.float64).reshape(-1)
    return K, dist

def read_vec(xml_path: Path, key: str) -> np.ndarray:
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

def build_ground_homography_worldXY_to_image(K, rvec, tvec):
    """
    z=0 平面单应： s [u v 1]^T = K [r1 r2 t] [X Y 1]^T
    X,Y,t 必须同单位（这里统一用 cm）
    """
    R, _ = cv2.Rodrigues(rvec)
    r1 = R[:, 0:1]          # 3x1
    r2 = R[:, 1:2]          # 3x1
    Rt = np.concatenate([r1, r2, tvec], axis=1)  # 3x3
    H = K @ Rt
    return H

def bev_pixel_to_world_xy_cm(u_bev, v_bev):
    # u_bev: 0..BEV_W-1 -> X
    # v_bev: 0..BEV_H-1 -> Y
    X = ORIG_X_CM + u_bev * STEP_CM
    Y = ORIG_Y_CM + v_bev * STEP_CM
    return X, Y

def build_T_bev_to_world():
    # [X,Y,1]^T = T * [u,v,1]^T
    return np.array([
        [STEP_CM, 0.0, ORIG_X_CM],
        [0.0, STEP_CM, ORIG_Y_CM],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

def warp_by_homography(img, K, dist, rvec, tvec):
    """
    路径 A：homography + warpPerspective
    关键点：OpenCV warpPerspective 默认期望输入的是 “src->dst”的 H，
    它内部会用逆映射做采样。
    我们更容易构造的是 M: (BEV -> Image)：
        p_img ~ H_w2i * T_bev2w * p_bev
    所以真正喂给 warpPerspective 的应该是 Image->BEV：
        H_img2bev = inv(M)
    """
    H_w2i = build_ground_homography_worldXY_to_image(K, rvec, tvec)  # world(XY)->img
    T_bev2w = build_T_bev_to_world()                                  # bev->world
    M_bev2img = H_w2i @ T_bev2w                                       # bev->img

    # === 关键修复：取逆，得到 image->bev ===
    H_img2bev = np.linalg.inv(M_bev2img)

    bev = cv2.warpPerspective(
        img,
        H_img2bev,
        dsize=(BEV_W, BEV_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return bev, M_bev2img

def warp_by_projectpoints_remap(img, K, dist, rvec, tvec):
    """
    路径 B：按 WILDTRACK-toolkit 思路：
    生成地面网格 (X,Y,0)，用 projectPoints 投到图像得到 (u,v)，再 remap 采样。
    这是最“没歧义”的实现，用来交叉验证 homography 是否写对。
    """
    # 生成 BEV 平面每个像素对应的世界点（cm）
    us = np.arange(BEV_W, dtype=np.float32)
    vs = np.arange(BEV_H, dtype=np.float32)
    U, V = np.meshgrid(us, vs)  # shape: (H, W)

    X = ORIG_X_CM + U * STEP_CM
    Y = ORIG_Y_CM + V * STEP_CM
    Z = np.full_like(X, Z_CM)

    Pw = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)  # (H*W,3)

    # projectPoints 输入单位必须与 tvec 一致（这里都是 cm）
    imgpts, _ = cv2.projectPoints(Pw.reshape(-1, 1, 3), rvec, tvec, K, dist)
    uv = imgpts.reshape(-1, 2).astype(np.float32)
    map_x = uv[:, 0].reshape(BEV_H, BEV_W)
    map_y = uv[:, 1].reshape(BEV_H, BEV_W)

    bev = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return bev

def load_image(frame_stem: str, view_num: int):
    cam_folder = f"C{view_num+1}"
    img_root = DATA_ROOT / "Image_subsets" / cam_folder
    img_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_root / f"{frame_stem}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        raise FileNotFoundError(f"Image not found: {img_root}/{frame_stem}.[png/jpg]")
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")
    return img, img_path

def load_calib(view_num: int):
    cam_name = CAM_NAMES[view_num]
    calib_root = DATA_ROOT / "calibrations"
    intr_path = calib_root / "intrinsic_original" / f"intr_{cam_name}.xml"
    extr_path = calib_root / "extrinsic" / f"extr_{cam_name}.xml"
    if not intr_path.exists():
        raise FileNotFoundError(f"Missing intrinsic: {intr_path}")
    if not extr_path.exists():
        raise FileNotFoundError(f"Missing extrinsic: {extr_path}")
    K, dist = read_intrinsics(intr_path)
    rvec = read_vec(extr_path, "rvec").reshape(3, 1)
    tvec = read_vec(extr_path, "tvec").reshape(3, 1)
    return cam_name, K, dist, rvec, tvec, intr_path, extr_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default="00000000")
    parser.add_argument("--view", type=int, default=0, help="0..6 maps to C1..C7")
    parser.add_argument("--method", type=str, default="both", choices=["H", "remap", "both"])
    args = parser.parse_args()

    view = args.view
    cam_folder = f"C{view+1}"
    cam_name, K, dist, rvec, tvec, intr_path, extr_path = load_calib(view)
    img, img_path = load_image(args.frame, view)

    out_dir = OUT_DIR / "bev_warp_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] img:", img_path)
    print("[INFO] intr:", intr_path)
    print("[INFO] extr:", extr_path)
    print("[INFO] bev grid:", f"{BEV_H}x{BEV_W}, origin=({ORIG_X_CM},{ORIG_Y_CM})cm step={STEP_CM}cm z=0")

    if args.method in ("H", "both"):
        bev_H, M_bev2img = warp_by_homography(img, K, dist, rvec, tvec)
        out_path = out_dir / f"{args.frame}_{cam_folder}_{cam_name}_bev_H.png"
        cv2.imwrite(str(out_path), bev_H)
        print("[OK] saved", out_path)

    if args.method in ("remap", "both"):
        bev_R = warp_by_projectpoints_remap(img, K, dist, rvec, tvec)
        out_path = out_dir / f"{args.frame}_{cam_folder}_{cam_name}_bev_remap.png"
        cv2.imwrite(str(out_path), bev_R)
        print("[OK] saved", out_path)

if __name__ == "__main__":
    main()
