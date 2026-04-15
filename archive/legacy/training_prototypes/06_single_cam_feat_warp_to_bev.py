import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms


try:
    from configs.config import DATA_ROOT, OUT_DIR
except Exception:
    BASE = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE / "wildtrack"
    OUT_DIR = BASE / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== WILDTRACK 地面网格（cm），与你 scale=100 的口径一致 =====
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

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, img_path


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


def build_bev_world_points_cm():
    us = np.arange(BEV_W, dtype=np.float32)
    vs = np.arange(BEV_H, dtype=np.float32)
    U, V = np.meshgrid(us, vs)  # (H,W)

    X = ORIG_X_CM + U * STEP_CM
    Y = ORIG_Y_CM + V * STEP_CM
    Z = np.full_like(X, Z_CM)

    Pw = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)  # (H*W,3)
    return Pw


def project_to_feature_coords(Pw_cm, K_feat, dist, rvec, tvec):
    """
    直接用缩放后的 K_feat 投影到“特征图像素坐标”(uf,vf)
    """
    imgpts, _ = cv2.projectPoints(Pw_cm.reshape(-1, 1, 3), rvec, tvec, K_feat, dist)
    uv = imgpts.reshape(-1, 2).astype(np.float32)
    return uv


def camera_depth_z(Pw_cm, rvec, tvec):
    """
    计算每个世界点在相机坐标的 z，用于过滤 z<=0 的点（这些点投影会出怪值）
    """
    R, _ = cv2.Rodrigues(rvec)
    Xc = (R @ Pw_cm.T) + tvec  # (3,N)
    z = Xc[2, :].astype(np.float64)
    return z


def uv_to_grid_sample(uv_feat, Wf, Hf, z_mask=None, align_corners=False):
    """
    uv_feat: (N,2) in feature pixel coords
    输出 grid: (1, BEV_H, BEV_W, 2) in [-1,1]
    对越界或 z<=0 的点，直接把 grid 设为 2.0（保证采样为 0）
    """
    uf = uv_feat[:, 0]
    vf = uv_feat[:, 1]

    valid = (uf >= 0) & (uf <= (Wf - 1)) & (vf >= 0) & (vf <= (Hf - 1))
    if z_mask is not None:
        valid = valid & z_mask

    # 先给全部点赋一个“肯定越界”的值
    x = np.full_like(uf, 2.0, dtype=np.float32)
    y = np.full_like(vf, 2.0, dtype=np.float32)

    ufv = uf[valid]
    vfv = vf[valid]

    if align_corners:
        x[valid] = 2.0 * (ufv / max(Wf - 1, 1)) - 1.0
        y[valid] = 2.0 * (vfv / max(Hf - 1, 1)) - 1.0
    else:
        # PyTorch 默认 align_corners=False 的坐标定义（MVDet 实现通常也用默认）
        x[valid] = (2.0 * ufv + 1.0) / max(Wf, 1) - 1.0
        y[valid] = (2.0 * vfv + 1.0) / max(Hf, 1) - 1.0

    grid = np.stack([x, y], axis=-1).reshape(BEV_H, BEV_W, 2).astype(np.float32)
    return grid, float(valid.mean())


class ResNet18Backbone(torch.nn.Module):
    """
    输出 stride=8 的特征图（到 layer2 为止）
    """
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # stride 4
        self.layer1 = m.layer1  # stride 4
        self.layer2 = m.layer2  # stride 8

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def save_energy_map(bev_feat: torch.Tensor, out_path: Path):
    with torch.no_grad():
        e = torch.sqrt(torch.sum(bev_feat[0] ** 2, dim=0))  # (Hb,Wb)
        e = e.cpu().numpy()

        # 只用非零区域做归一化，避免“99%是0”时的尴尬
        nz = e[e > 1e-9]
        if nz.size == 0:
            img = np.zeros_like(e, dtype=np.uint8)
            cv2.imwrite(str(out_path), img)
            return

        scale = np.percentile(nz, 99)
        e = e / (scale + 1e-6)
        e = np.clip(e, 0, 1)
        img = (e * 255).astype(np.uint8)
        cv2.imwrite(str(out_path), img)


def main(frame: str, view: int, device: str):
    assert 0 <= view <= 6
    cam_folder = f"C{view+1}"

    cam_name, K, dist, rvec, tvec, intr_path, extr_path = load_calib(view)
    rgb, img_path = load_image(frame, view)
    H, W = rgb.shape[:2]

    print("[INFO] img:", img_path)
    print("[INFO] intr:", intr_path)
    print("[INFO] extr:", extr_path)
    print("[INFO] image size:", (W, H)
          )

    # 1) backbone 特征
    backbone = ResNet18Backbone().to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = preprocess(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = backbone(x)  # (1,C,Hf,Wf)

    _, C, Hf, Wf = feat.shape
    stride_x = W / float(Wf)
    stride_y = H / float(Hf)
    print("[INFO] feat shape:", (C, Hf, Wf))
    print("[INFO] approx stride:", (stride_x, stride_y))

    # 2) 缩放内参到特征图坐标系（MVDet常用做法）
    #    直接把 K 的前两行除以 stride（假设 stride_x≈stride_y）
    #    如果两者略不同，也分别缩放
    Kf = K.copy()
    Kf[0, :] /= stride_x
    Kf[1, :] /= stride_y

    # 3) 构造 BEV 网格世界点（cm）
    Pw = build_bev_world_points_cm()  # (Hb*Wb,3)

    # 4) 过滤 z<=0（这些点投影会发散/产生inf）
    z = camera_depth_z(Pw, rvec, tvec)  # (N,)
    z_mask = z > 1e-6

    # 5) 直接投影到“特征图像素坐标”
    uv_feat = project_to_feature_coords(Pw, Kf, dist, rvec, tvec)  # (N,2)

    # 6) 转 grid_sample 网格（默认 align_corners=False）
    grid_np, valid_ratio = uv_to_grid_sample(uv_feat, Wf, Hf, z_mask=z_mask, align_corners=False)
    print(f"[DEBUG] grid valid ratio (in-range & z>0) = {valid_ratio:.4f}")

    grid = torch.from_numpy(grid_np).unsqueeze(0).to(device)  # (1,Hb,Wb,2)

    # 7) warp 特征到 BEV
    bev_feat = F.grid_sample(
        feat, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )

    out_dir = OUT_DIR / "bev_feat_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_energy = out_dir / f"{frame}_{cam_folder}_{cam_name}_bev_feat_energy.png"
    save_energy_map(bev_feat, out_energy)
    print("[OK] saved", out_energy)

    out_txt = out_dir / f"{frame}_{cam_folder}_{cam_name}_info.txt"
    out_txt.write_text(
        "\n".join([
            f"image_wh = {(W, H)}",
            f"feat_wh  = {(Wf, Hf)}",
            f"stride_xy ~ {(stride_x, stride_y)}",
            f"grid_valid_ratio = {valid_ratio}",
            f"bev_wh   = {(BEV_W, BEV_H)}",
            f"origin_cm= {(ORIG_X_CM, ORIG_Y_CM)}",
            f"step_cm  = {STEP_CM}",
            f"z_cm     = {Z_CM}",
            f"intr     = {intr_path}",
            f"extr     = {extr_path}",
        ]),
        encoding="utf-8"
    )
    print("[OK] saved", out_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default="00000000")
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args.frame, args.view, args.device)
