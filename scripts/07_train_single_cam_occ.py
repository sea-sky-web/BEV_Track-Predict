import argparse
from pathlib import Path
import json
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    from configs.config import DATA_ROOT, OUT_DIR
except Exception:
    BASE = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE / "wildtrack"
    OUT_DIR = BASE / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== WILDTRACK ground grid base definition (cm) =====
ORIG_X_CM = -300.0
ORIG_Y_CM = -900.0
STEP_CM_BASE = 2.5
BEV_W_BASE = 480
BEV_H_BASE = 1440
Z_CM = 0.0

CAM_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]


# ----------------- IO: calib -----------------
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


def load_calib_by_view(view_num: int):
    cam_name = CAM_NAMES[view_num]
    calib_root = DATA_ROOT / "calibrations"
    intr_path = calib_root / "intrinsic_original" / f"intr_{cam_name}.xml"
    extr_path = calib_root / "extrinsic" / f"extr_{cam_name}.xml"
    K, dist = read_intrinsics(intr_path)
    rvec = read_vec(extr_path, "rvec").reshape(3, 1)
    tvec = read_vec(extr_path, "tvec").reshape(3, 1)
    return cam_name, K, dist, rvec, tvec


# ----------------- label: occupancy heatmap -----------------
def gaussian_2d(H, W, cx, cy, sigma=1.5):
    radius = int(3 * sigma)
    x0, x1 = max(0, cx - radius), min(W - 1, cx + radius)
    y0, y1 = max(0, cy - radius), min(H - 1, cy + radius)

    xs = np.arange(x0, x1 + 1)
    ys = np.arange(y0, y1 + 1)
    X, Y = np.meshgrid(xs, ys)

    g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma * sigma))
    return x0, y0, g.astype(np.float32)


def build_occ_label_from_json(json_path: Path, bev_down: int, sigma: float):
    Hb = BEV_H_BASE // bev_down
    Wb = BEV_W_BASE // bev_down
    label = np.zeros((Hb, Wb), dtype=np.float32)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for obj in data:
        pos_id = obj.get("positionID", None)
        if pos_id is None:
            continue
        pos_id = int(pos_id)

        ix = pos_id % BEV_W_BASE
        iy = pos_id // BEV_W_BASE
        ix_ds = ix // bev_down
        iy_ds = iy // bev_down

        if 0 <= ix_ds < Wb and 0 <= iy_ds < Hb:
            x0, y0, g = gaussian_2d(Hb, Wb, ix_ds, iy_ds, sigma=sigma)
            h, w = g.shape
            label[y0:y0 + h, x0:x0 + w] = np.maximum(label[y0:y0 + h, x0:x0 + w], g)

    return label


# ----------------- geometry: build grid for grid_sample -----------------
def build_bev_world_points_cm(bev_down: int):
    step = STEP_CM_BASE * bev_down
    Wb = BEV_W_BASE // bev_down
    Hb = BEV_H_BASE // bev_down

    us = np.arange(Wb, dtype=np.float32)
    vs = np.arange(Hb, dtype=np.float32)
    U, V = np.meshgrid(us, vs)

    X = ORIG_X_CM + U * step
    Y = ORIG_Y_CM + V * step
    Z = np.full_like(X, Z_CM)
    Pw = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)
    return Pw, Hb, Wb


def camera_depth_z(Pw_cm, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    Xc = (R @ Pw_cm.T) + tvec
    return Xc[2, :].astype(np.float64)


def project_to_feature_coords(Pw_cm, K_feat, dist, rvec, tvec):
    imgpts, _ = cv2.projectPoints(Pw_cm.reshape(-1, 1, 3), rvec, tvec, K_feat, dist)
    return imgpts.reshape(-1, 2).astype(np.float32)


def uv_to_grid_sample(uv_feat, Wf, Hf, valid_mask=None, align_corners=False):
    uf = uv_feat[:, 0]
    vf = uv_feat[:, 1]

    valid = (uf >= 0) & (uf <= (Wf - 1)) & (vf >= 0) & (vf <= (Hf - 1))
    if valid_mask is not None:
        valid = valid & valid_mask

    x = np.full_like(uf, 2.0, dtype=np.float32)
    y = np.full_like(vf, 2.0, dtype=np.float32)

    ufv = uf[valid]
    vfv = vf[valid]

    if align_corners:
        x[valid] = 2.0 * (ufv / max(Wf - 1, 1)) - 1.0
        y[valid] = 2.0 * (vfv / max(Hf - 1, 1)) - 1.0
    else:
        x[valid] = (2.0 * ufv + 1.0) / max(Wf, 1) - 1.0
        y[valid] = (2.0 * vfv + 1.0) / max(Hf, 1) - 1.0

    return np.stack([x, y], axis=-1).astype(np.float32), float(valid.mean())


# ----------------- dataset -----------------
class WildtrackSingleCamDataset(Dataset):
    def __init__(self, view_num: int, bev_down: int, sigma: float, max_frames: int = 0):
        self.view = view_num
        self.cam_folder = f"C{view_num+1}"
        self.ann_dir = DATA_ROOT / "annotations_positions"
        self.img_dir = DATA_ROOT / "Image_subsets" / self.cam_folder
        self.files = sorted(self.ann_dir.glob("*.json"))
        if max_frames > 0:
            self.files = self.files[:max_frames]
        assert len(self.files) > 0, f"No json files in {self.ann_dir}"

        self.bev_down = bev_down
        self.sigma = sigma

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        json_path = self.files[idx]
        stem = json_path.stem

        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            return self.__getitem__((idx + 1) % len(self.files))

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.preprocess(rgb)

        y = build_occ_label_from_json(json_path, self.bev_down, self.sigma)
        y = torch.from_numpy(y).unsqueeze(0)

        return x, y, stem


# ----------------- model -----------------
class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class BEVOccHead(nn.Module):
    def __init__(self, in_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class SingleCamMVDetMini(nn.Module):
    def __init__(self, view_num: int, bev_down: int, device: torch.device):
        super().__init__()
        self.view = view_num
        self.bev_down = bev_down
        self.device_ = device

        self.cam_name, self.K, self.dist, self.rvec, self.tvec = load_calib_by_view(view_num)

        self.backbone = ResNet18Backbone()
        self.head = BEVOccHead(128)

        self._grid = None

    def build_grid(self, img_wh, feat_wh):
        W, H = img_wh
        Wf, Hf = feat_wh

        stride_x = W / float(Wf)
        stride_y = H / float(Hf)

        Kf = self.K.copy()
        Kf[0, :] /= stride_x
        Kf[1, :] /= stride_y

        Pw, Hb, Wb = build_bev_world_points_cm(self.bev_down)
        z = camera_depth_z(Pw, self.rvec, self.tvec)
        z_mask = z > 1e-6

        uv_feat = project_to_feature_coords(Pw, Kf, self.dist, self.rvec, self.tvec)
        grid_flat, valid_ratio = uv_to_grid_sample(uv_feat, Wf, Hf, valid_mask=z_mask, align_corners=False)

        grid = grid_flat.reshape(Hb, Wb, 2)
        grid_t = torch.from_numpy(grid).unsqueeze(0).to(self.device_)

        self._grid = grid_t
        print(f"[GRID] built for {self.cam_name}: feat={feat_wh}, bev={Wb}x{Hb}, valid_ratio={valid_ratio:.4f}")

    def forward(self, x):
        B, _, H, W = x.shape
        feat = self.backbone(x)
        _, _, Hf, Wf = feat.shape

        if self._grid is None:
            self.build_grid(img_wh=(W, H), feat_wh=(Wf, Hf))

        bev_feat = F.grid_sample(
            feat, self._grid.expand(B, -1, -1, -1),
            mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return self.head(bev_feat)


# ----------------- save utils -----------------
def save_heatmap(t: torch.Tensor, out_path: Path):
    a = t.detach().cpu().numpy()
    a = a / (np.max(a) + 1e-6)
    img = (a * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), img)


# ----------------- main -----------------
def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You requested --device cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be one of: cuda, cpu, auto")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--bev_down", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)

    # GPU 确认信息（关键：以后你看日志就知道是不是GPU）
    print("[DEV] torch.cuda.is_available =", torch.cuda.is_available())
    print("[DEV] selected device =", device)
    if device.type == "cuda":
        print("[DEV] gpu name =", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True  # 训练时加速

    ds = WildtrackSingleCamDataset(args.view, args.bev_down, args.sigma, max_frames=args.max_frames)

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = SingleCamMVDetMini(args.view, args.bev_down, device).to(device)
    model.train()

    # 再次确认模型在什么设备上
    pdev = next(model.parameters()).device
    print("[DEV] model param device =", pdev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pos_weight = torch.tensor([20.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out_dir = OUT_DIR / "train_single_cam_occ"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for ep in range(args.epochs):
        for x, y, stem in dl:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % args.log_every == 0:
                print(f"[ep {ep} step {step}] loss={loss.item():.4f}")
                with torch.no_grad():
                    pred = torch.sigmoid(logits[0, 0])
                    gt = y[0, 0]
                    save_heatmap(pred, out_dir / f"{stem[0]}_pred.png")
                    save_heatmap(gt,   out_dir / f"{stem[0]}_gt.png")

            step += 1

    torch.save(model.state_dict(), out_dir / "model_single_cam.pth")
    print("[OK] saved model:", out_dir / "model_single_cam.pth")


if __name__ == "__main__":
    main()
