# scripts/08_train_multicam_mvdet_style_v3.py
# 对齐 MVDet 源码版本：
# 1) Loss: GaussianMSE (adaptive_max_pool2d -> gaussian conv -> mse)
#    total = bev_loss + alpha * (per_view_loss / V)
# 2) Projection: use homography imgcoord2worldgrid, then warp_perspective(img_feature, H, reducedgrid_shape)
# 3) Grid shape: worldgrid is (NB_HEIGHT, NB_WIDTH) = (1440,480)  => reduced (360,120) for bev_down=4
#
# 依赖：numpy, opencv-python, pillow, torch, torchvision

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision


# ---- Wildtrack rectangles.pom constants (fallback) ----
ORIGINE_X_M = -3.0
ORIGINE_Y_M = -9.0
NB_WIDTH = 480     # x bins
NB_HEIGHT = 1440   # y bins
WIDTH_M = 12.0
STEP_M = WIDTH_M / NB_WIDTH  # 0.025m

IMG_ORI_W, IMG_ORI_H = 1920, 1080
CAM_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]


def parse_rectangles_pom(pom_path: Path) -> Dict[str, float]:
    if not pom_path.exists():
        return {}
    txt = pom_path.read_text(encoding="utf-8", errors="ignore")
    kv = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                kv[k] = float(v)
            except:
                pass
    return kv


def read_intrinsics(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    K = np.array(K, dtype=np.float64)
    dist = np.array(dist, dtype=np.float64).reshape(-1)
    return K, dist


def read_opencv_xml_vec(xml_path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    node = fs.getNode(key)
    vals = [node.at(i).real() for i in range(node.size())]
    fs.release()
    return np.array(vals, dtype=np.float64)


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def build_gaussian_kernel_2d(ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
    assert ksize % 2 == 1, "ksize must be odd"
    r = ksize // 2
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    ys = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    g = g / (g.max().clamp_min(1e-12))
    # shape (1,1,K,K)
    return g.view(1, 1, ksize, ksize)


class GaussianMSE(nn.Module):
    """
    对齐 MVDet gaussian_mse.py 的逻辑：
    - target -> adaptive_max_pool2d 到 pred 的空间尺寸
    - 用 gaussian kernel 做 conv2d 平滑
    - MSE(pred, target_smooth)
    支持多通道：对每个 channel 独立卷积（groups=C）
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B,C,H,W) - 建议已经在 [0,1]（即 sigmoid 后）
        target: (B,C,Ht,Wt) - 二值热图（点为1）
        kernel: (1,1,K,K) gaussian
        """
        B, C, H, W = pred.shape
        target = F.adaptive_max_pool2d(target, output_size=(H, W))

        # per-channel gaussian conv
        # reshape to (B*C,1,H,W)
        tgt = target.reshape(B * C, 1, H, W)
        k = kernel.to(dtype=tgt.dtype, device=tgt.device)
        pad = (k.shape[-1] - 1) // 2
        tgt = F.conv2d(tgt, k, padding=pad)
        tgt = tgt.reshape(B, C, H, W)

        return F.mse_loss(pred, tgt)


def warp_perspective_torch(src: torch.Tensor, M_src2dst: torch.Tensor, dsize: Tuple[int, int]) -> torch.Tensor:
    """
    纯 PyTorch 实现 warp_perspective，语义对齐 OpenCV/Kornia：
    - 给定 M: src(x,y,1) -> dst(x,y,1)
    - 采样时用 inverse mapping: src = inv(M) @ dst
    src: (B,C,Hs,Ws)
    M_src2dst: (B,3,3)
    dsize: (Hd, Wd)
    return: (B,C,Hd,Wd)
    """
    B, C, Hs, Ws = src.shape
    Hd, Wd = dsize
    device = src.device
    dtype = src.dtype

    # dst pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(Hd, device=device, dtype=dtype),
        torch.arange(Wd, device=device, dtype=dtype),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    dst_h = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # (3, Hd*Wd)
    dst_h = dst_h.unsqueeze(0).expand(B, -1, -1)               # (B,3,N)

    M_inv = torch.inverse(M_src2dst.to(dtype=dtype, device=device))  # (B,3,3)
    src_h = M_inv @ dst_h                                            # (B,3,N)
    x = src_h[:, 0] / src_h[:, 2].clamp_min(1e-6)
    y = src_h[:, 1] / src_h[:, 2].clamp_min(1e-6)

    # normalize to [-1,1]
    x_norm = 2.0 * (x / (Ws - 1.0)) - 1.0
    y_norm = 2.0 * (y / (Hs - 1.0)) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(B, Hd, Wd, 2)  # (B,Hd,Wd,2)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out


def compute_valid_ratio_from_homography(M_src2dst: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> float:
    """
    用 homography 估计有效采样比例：dst 网格映射回 src，看落在 src 范围内的比例
    """
    Hs, Ws = src_hw
    Hd, Wd = dst_hw
    try:
        M_inv = np.linalg.inv(M_src2dst)
    except np.linalg.LinAlgError:
        return 0.0

    xs = np.arange(Wd, dtype=np.float64)
    ys = np.arange(Hd, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (Hd,Wd)
    ones = np.ones_like(xx)
    dst = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T  # (3,N)

    src = M_inv @ dst
    x = src[0] / (src[2] + 1e-9)
    y = src[1] / (src[2] + 1e-9)

    valid = (x >= 0) & (x <= (Ws - 1)) & (y >= 0) & (y <= (Hs - 1))
    return float(valid.mean())


def make_worldgrid2worldcoord_mat(origin_x: float, origin_y: float, step: float) -> np.ndarray:
    """
    worldgrid (x_idx, y_idx, 1) -> worldcoord (X, Y, 1)
    使用 cell center：origin + (idx + 0.5) * step
    """
    ox = origin_x + 0.5 * step
    oy = origin_y + 0.5 * step
    return np.array([
        [step, 0.0, ox],
        [0.0, step, oy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def build_mvdet_proj_mat(
    K_feat: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    worldgrid2worldcoord: np.ndarray,
) -> np.ndarray:
    """
    对齐 MVDet 源码：
    worldcoord2imgcoord = K @ np.delete(extrinsic, 2, 1)
    worldgrid2imgcoord  = worldcoord2imgcoord @ worldgrid2worldcoord
    imgcoord2worldgrid  = inv(worldgrid2imgcoord)
    proj_mat            = permutation @ imgcoord2worldgrid
    """
    extr = np.concatenate([R, t.reshape(3, 1)], axis=1)  # (3,4) world->cam
    extr_3x3 = np.delete(extr, 2, 1)                    # remove Z column => (3,3): [r1 r2 t]
    worldcoord2imgcoord = K_feat @ extr_3x3             # (3,3)
    worldgrid2imgcoord = worldcoord2imgcoord @ worldgrid2worldcoord  # (3,3)
    imgcoord2worldgrid = np.linalg.inv(worldgrid2imgcoord)

    permutation = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1]], dtype=np.float64)
    proj = permutation @ imgcoord2worldgrid
    return proj


def save_heat_png(path: Path, arr: np.ndarray):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


class WildtrackMVDetDataset(Dataset):
    """
    输出：
      - x_views: (V,3,Hi,Wi)
      - map_gt:  (1, NB_HEIGHT, NB_WIDTH)  二值点图（full grid）
      - imgs_gt: (V,2,Hf,Wf) head/foot 二值点图（feature plane）
    """
    def __init__(
        self,
        data_root: Path,
        views: List[int],
        max_frames: int,
        img_hw: Tuple[int, int],
        feat_hw: Tuple[int, int],
        bev_down: int,
        person_h_m: float,
        unit_scale: float,
        calib_cache: Dict[int, Dict[str, Any]],
    ):
        self.data_root = data_root
        self.views = views
        self.Hi, self.Wi = img_hw
        self.Hf, self.Wf = feat_hw
        self.bev_down = bev_down
        self.person_h = person_h_m * unit_scale
        self.unit_scale = unit_scale
        self.calib_cache = calib_cache

        self.ann_dir = data_root / "annotations_positions"
        assert self.ann_dir.exists()

        self.img_dirs = []
        for v in views:
            p = data_root / "Image_subsets" / f"C{v+1}"
            assert p.exists(), p
            self.img_dirs.append(p)

        self.ann_files = sorted(self.ann_dir.glob("*.json"))
        if max_frames > 0:
            self.ann_files = self.ann_files[:max_frames]
        assert len(self.ann_files) > 0

        # ImageNet normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # rectangles constants
        self.nb_w = NB_WIDTH
        self.nb_h = NB_HEIGHT

        # world grid params in chosen unit
        step = STEP_M * unit_scale
        self.ox = ORIGINE_X_M * unit_scale
        self.oy = ORIGINE_Y_M * unit_scale
        self.step = step

    def __len__(self):
        return len(self.ann_files)

    def _load_image(self, img_dir: Path, stem: str) -> Image.Image:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"missing image for {stem} in {img_dir}")

    def __getitem__(self, idx: int):
        ann_path = self.ann_files[idx]
        stem = ann_path.stem
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        # images
        xs = []
        for img_dir in self.img_dirs:
            img = self._load_image(img_dir, stem).resize((self.Wi, self.Hi), Image.BILINEAR)
            x = torch.from_numpy(np.array(img, dtype=np.uint8)).float() / 255.0
            x = x.permute(2, 0, 1)
            x = (x - self.mean) / self.std
            xs.append(x)
        x_views = torch.stack(xs, dim=0)  # (V,3,Hi,Wi)

        # map_gt full grid: (1, NB_HEIGHT, NB_WIDTH)   (row=y, col=x)
        map_gt = torch.zeros((1, self.nb_h, self.nb_w), dtype=torch.float32)

        # imgs_gt per view at feature plane: (V,2,Hf,Wf)
        imgs_gt = torch.zeros((len(self.views), 2, self.Hf, self.Wf), dtype=torch.float32)

        for obj in data:
            pos_id = obj.get("positionID", None)
            if pos_id is None:
                continue
            pos_id = int(pos_id)
            ix = pos_id % self.nb_w           # x index (0..479)
            iy = pos_id // self.nb_w          # y index (0..1439)

            # full-grid point
            if 0 <= iy < self.nb_h and 0 <= ix < self.nb_w:
                map_gt[0, iy, ix] = 1.0

            # world coord (X,Y) in chosen unit
            Xw = self.ox + (ix + 0.5) * self.step
            Yw = self.oy + (iy + 0.5) * self.step

            Pw_foot = np.array([Xw, Yw, 0.0], dtype=np.float64).reshape(3, 1)
            Pw_head = np.array([Xw, Yw, self.person_h], dtype=np.float64).reshape(3, 1)

            for vi, v in enumerate(self.views):
                calib = self.calib_cache[v]
                Kf = calib["K_feat"]
                R = calib["R"]
                t = calib["t"]

                # project (ignore distortion, align MVDet homography assumption)
                def proj(Pw):
                    Pc = R @ Pw + t
                    z = float(Pc[2, 0])
                    if z <= 1e-6:
                        return None
                    u = (Kf[0, 0] * (Pc[0, 0] / z) + Kf[0, 2])
                    v_ = (Kf[1, 1] * (Pc[1, 0] / z) + Kf[1, 2])
                    return float(u), float(v_)

                p_head = proj(Pw_head)
                p_foot = proj(Pw_foot)
                if p_head is not None:
                    u, v_ = p_head
                    x = int(round(u))
                    y = int(round(v_))
                    if 0 <= x < self.Wf and 0 <= y < self.Hf:
                        imgs_gt[vi, 0, y, x] = 1.0
                if p_foot is not None:
                    u, v_ = p_foot
                    x = int(round(u))
                    y = int(round(v_))
                    if 0 <= x < self.Wf and 0 <= y < self.Hf:
                        imgs_gt[vi, 1, y, x] = 1.0

        return stem, x_views, map_gt, imgs_gt


class ResNet50Stride8Trunk(nn.Module):
    """
    用 resnet50 + dilation 输出 stride=8（更接近 MVDet 的容量/语义）
    """
    def __init__(self, pretrained: bool, out_ch: int = 512):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.reduce = nn.Conv2d(2048, out_ch, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.reduce(x)
        return x


class BEVHeadDilated(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=4, dilation=4),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class ImgHeadFoot(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class MVDetLikeNet(nn.Module):
    """
    输出：
      - map_res:  (B,1,Hb,Wb)  -> sigmoid 后进 GaussianMSE
      - imgs_res: (B,V,2,Hf,Wf)-> sigmoid 后进 GaussianMSE
    """
    def __init__(
        self,
        num_views: int,
        proj_mats: torch.Tensor,    # (V,3,3) src(feature)->dst(worldgrid)
        reduced_hw: Tuple[int, int],
        feat_hw: Tuple[int, int],
        feat_ch: int = 512,
        pretrained: bool = True,
        add_coord: bool = True,
    ):
        super().__init__()
        self.V = num_views
        self.Hb, self.Wb = reduced_hw
        self.Hf, self.Wf = feat_hw
        self.add_coord = add_coord

        self.backbone = ResNet50Stride8Trunk(pretrained=pretrained, out_ch=feat_ch)
        self.img_head = ImgHeadFoot(in_ch=feat_ch, mid_ch=128)

        self.proj_mats = nn.Parameter(proj_mats, requires_grad=False)  # (V,3,3)

        in_bev = num_views * feat_ch
        if add_coord:
            in_bev += 2
            xs = torch.linspace(-1, 1, self.Hb).view(self.Hb, 1).expand(self.Hb, self.Wb)
            ys = torch.linspace(-1, 1, self.Wb).view(1, self.Wb).expand(self.Hb, self.Wb)
            coord = torch.stack([ys, xs], dim=0).unsqueeze(0)  # (1,2,Hb,Wb) note: (x,y) ordering not critical
            self.coord = nn.Parameter(coord, requires_grad=False)
        else:
            self.coord = None

        self.bev_head = BEVHeadDilated(in_ch=in_bev, mid_ch=256)

    def forward(self, x_views: torch.Tensor):
        """
        x_views: (B,V,3,Hi,Wi)
        """
        B, V, _, _, _ = x_views.shape
        feats_bev = []
        imgs_logits = []

        for vi in range(V):
            f = self.backbone(x_views[:, vi])  # (B,C,Hi/8,Wi/8)
            f = F.interpolate(f, size=(self.Hf, self.Wf), mode="bilinear", align_corners=False)

            img_logit = self.img_head(f)  # (B,2,Hf,Wf)
            imgs_logits.append(img_logit)

            M = self.proj_mats[vi].unsqueeze(0).expand(B, -1, -1)  # (B,3,3)
            bev = warp_perspective_torch(f, M, dsize=(self.Hb, self.Wb))  # (B,C,Hb,Wb)
            feats_bev.append(bev)

        imgs_logits = torch.stack(imgs_logits, dim=1)  # (B,V,2,Hf,Wf)
        bev_cat = torch.cat(feats_bev, dim=1)          # (B,V*C,Hb,Wb)

        if self.add_coord:
            coord = self.coord.to(bev_cat.device).expand(B, -1, -1, -1)
            bev_cat = torch.cat([bev_cat, coord], dim=1)

        map_logits = self.bev_head(bev_cat)  # (B,1,Hb,Wb)
        return map_logits, imgs_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="wildtrack")
    ap.add_argument("--views", type=str, default="0,1,2", help="e.g. 0,3,5")
    ap.add_argument("--drop_bad_views", action="store_true")
    ap.add_argument("--valid_thr", type=float, default=0.10)

    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)

    ap.add_argument("--bev_down", type=int, default=4)
    ap.add_argument("--feat_h", type=int, default=270)
    ap.add_argument("--feat_w", type=int, default=480)
    ap.add_argument("--img_h", type=int, default=720)
    ap.add_argument("--img_w", type=int, default=1280)

    ap.add_argument("--person_h", type=float, default=1.7)
    ap.add_argument("--alpha", type=float, default=1.0)

    ap.add_argument("--map_ksize", type=int, default=11)
    ap.add_argument("--map_sigma", type=float, default=2.5)
    ap.add_argument("--img_ksize", type=int, default=11)
    ap.add_argument("--img_sigma", type=float, default=2.0)

    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze_bn", action="store_true")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_lr", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_steps", type=str, default="0,200,1000")
    ap.add_argument("--fixed_stem", type=str, default="00000055")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path("outputs") / "train_multicam_mvdet_style_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device)
    print("[DEV] device =", dev, "cuda_available=", torch.cuda.is_available())
    if dev.type == "cuda":
        print("[DEV] gpu =", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # rectangles.pom (fallback if missing)
    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    nb_w = int(pom.get("NB_WIDTH", NB_WIDTH))
    nb_h = int(pom.get("NB_HEIGHT", NB_HEIGHT))
    origin_x_m = float(pom.get("ORIGINE_X", ORIGINE_X_M))
    origin_y_m = float(pom.get("ORIGINE_Y", ORIGINE_Y_M))
    step_m = float(pom.get("STEP", STEP_M))

    # reduced grid shape (MVDet worldgrid: N_row x N_col = NB_HEIGHT x NB_WIDTH)
    Hb = nb_h // args.bev_down
    Wb = nb_w // args.bev_down
    print(f"[CFG] views={args.views}, img={args.img_h}x{args.img_w}, feat={args.feat_h}x{args.feat_w}, bev(reduced)={Hb}x{Wb}, bev_down={args.bev_down}")

    # parse views
    views = [int(x.strip()) for x in args.views.split(",") if x.strip().isdigit()]
    assert len(views) > 0

    # load calib, decide unit_scale (m vs cm) by tvec magnitude
    calib_root = data_root / "calibrations"
    calib_cache: Dict[int, Dict[str, Any]] = {}
    t_norms = []
    for v in views:
        cam_name = CAM_NAMES[v]
        intr_path = calib_root / "intrinsic_original" / f"intr_{cam_name}.xml"
        extr_path = calib_root / "extrinsic" / f"extr_{cam_name}.xml"
        assert intr_path.exists(), intr_path
        assert extr_path.exists(), extr_path

        K0, dist = read_intrinsics(intr_path)
        rvec = read_opencv_xml_vec(extr_path, "rvec").reshape(3, 1)
        tvec = read_opencv_xml_vec(extr_path, "tvec").reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        t = tvec.astype(np.float64)

        calib_cache[v] = {"K0": K0, "R": R, "t": t, "cam": cam_name}
        t_norms.append(float(np.linalg.norm(t)))

    # heuristic: if t in [100..1000], likely cm; if step is 0.025m, convert world(m) -> cm by 100
    unit_scale = 1.0
    if step_m < 1.0 and np.median(t_norms) > 50.0:
        unit_scale = 100.0
    print(f"[UNIT] step={step_m} median||t||={np.median(t_norms):.2f} => unit_scale={unit_scale} (world coord will be in {'cm' if unit_scale==100 else 'm'})")

    # build projection matrices like MVDet (feature plane coordinate)
    Hf, Wf = args.feat_h, args.feat_w

    # scale K0 directly from original image to feature plane
    sx_f = Wf / IMG_ORI_W
    sy_f = Hf / IMG_ORI_H

    # worldgrid2worldcoord for reduced grid (with bev_down)
    step = (step_m * args.bev_down) * unit_scale
    ox = origin_x_m * unit_scale
    oy = origin_y_m * unit_scale
    worldgrid2worldcoord = make_worldgrid2worldcoord_mat(ox, oy, step)

    proj_mats = []
    kept_views = []
    for v in views:
        K0 = calib_cache[v]["K0"]
        R = calib_cache[v]["R"]
        t = calib_cache[v]["t"]

        K_feat = scale_intrinsics(K0, sx=sx_f, sy=sy_f)
        calib_cache[v]["K_feat"] = K_feat

        try:
            proj = build_mvdet_proj_mat(K_feat=K_feat, R=R, t=t, worldgrid2worldcoord=worldgrid2worldcoord)
        except np.linalg.LinAlgError:
            vr = 0.0
            proj = None

        if proj is None:
            print(f"[GRID] view={v} cam={calib_cache[v]['cam']} proj=singular valid_ratio=0.0000")
            if args.drop_bad_views:
                continue
            else:
                raise RuntimeError("Projection matrix singular. Fix calibration / unit.")
        else:
            vr = compute_valid_ratio_from_homography(proj, src_hw=(Hf, Wf), dst_hw=(Hb, Wb))
            print(f"[GRID] view={v} cam={calib_cache[v]['cam']} valid_ratio={vr:.4f}")

            if args.drop_bad_views and vr < args.valid_thr:
                print(f"[GRID] drop view={v} due to valid_ratio<{args.valid_thr}")
                continue

            proj_mats.append(torch.from_numpy(proj).float())
            kept_views.append(v)

    assert len(kept_views) > 0, "No valid views kept. Fix calibration/unit or lower valid_thr."
    print("[CFG] kept_views =", kept_views)

    proj_mats_t = torch.stack(proj_mats, dim=0).to(dev)  # (V,3,3)

    # dataset
    ds = WildtrackMVDetDataset(
        data_root=data_root,
        views=kept_views,
        max_frames=args.max_frames,
        img_hw=(args.img_h, args.img_w),
        feat_hw=(args.feat_h, args.feat_w),
        bev_down=args.bev_down,
        person_h_m=args.person_h,
        unit_scale=unit_scale,
        calib_cache=calib_cache,
    )

    def collate(batch):
        stems, x_views, map_gt, imgs_gt = zip(*batch)
        return list(stems), torch.stack(x_views, 0), torch.stack(map_gt, 0), torch.stack(imgs_gt, 0)

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        drop_last=True,
        collate_fn=collate,
    )
    print("[DATA] len(ds)=", len(ds), "len(loader)=", len(loader), "steps/epoch=", len(loader))

    # model
    model = MVDetLikeNet(
        num_views=len(kept_views),
        proj_mats=proj_mats_t,
        reduced_hw=(Hb, Wb),
        feat_hw=(args.feat_h, args.feat_w),
        feat_ch=512,
        pretrained=args.pretrained,
        add_coord=True,
    ).to(dev)

    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        print("[CFG] freeze_bn = True")

    # MVDet style optimizer
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=5e-4)
    steps_per_epoch = len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    criterion = GaussianMSE()

    save_steps = set(int(s.strip()) for s in args.save_steps.split(",") if s.strip().isdigit())

    # kernels
    map_kernel = build_gaussian_kernel_2d(args.map_ksize, args.map_sigma, device=dev)  # (1,1,K,K)
    img_kernel = build_gaussian_kernel_2d(args.img_ksize, args.img_sigma, device=dev)

    # fixed stem cache
    fixed_item = None
    if args.fixed_stem:
        ann = data_root / "annotations_positions" / f"{args.fixed_stem}.json"
        if ann.exists():
            for i in range(len(ds)):
                stem, x_views, map_gt, imgs_gt = ds[i]
                if stem == args.fixed_stem:
                    fixed_item = (stem, x_views, map_gt, imgs_gt)
                    print("[FIXED] using stem =", args.fixed_stem)
                    break

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and dev.type == "cuda"))

    global_step = 0
    model.train()

    for ep in range(args.epochs):
        for stems, x_views, map_gt, imgs_gt in loader:
            x_views = x_views.to(dev, non_blocking=True)     # (B,V,3,Hi,Wi)
            map_gt = map_gt.to(dev, non_blocking=True)       # (B,1,NBH,NBW)
            imgs_gt = imgs_gt.to(dev, non_blocking=True)     # (B,V,2,Hf,Wf)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                map_logits, imgs_logits = model(x_views)
                map_res = torch.sigmoid(map_logits)               # (B,1,Hb,Wb)
                imgs_res = torch.sigmoid(imgs_logits)             # (B,V,2,Hf,Wf)

                bev_loss = criterion(map_res, map_gt, map_kernel)

                per_view_loss = 0.0
                for vi in range(imgs_res.shape[1]):
                    # (B,2,Hf,Wf)
                    per_view_loss = per_view_loss + criterion(imgs_res[:, vi], imgs_gt[:, vi], img_kernel)

                per_view_loss = per_view_loss / float(imgs_res.shape[1])
                loss = bev_loss + args.alpha * per_view_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # IMPORTANT: scheduler after optimizer step (fix the warning)
            sched.step()

            if global_step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                with torch.no_grad():
                    # pos mse on BEV (only where pooled GT has signal)
                    pooled_gt = F.adaptive_max_pool2d(map_gt, output_size=map_res.shape[-2:])
                    pos_mask = pooled_gt > 0.1
                    pos_mse = ((map_res - pooled_gt) ** 2)[pos_mask].mean().item() if pos_mask.any() else float("nan")

                    # aux_pos_mse
                    aux_pos_mask = imgs_gt > 0.1
                    aux_pos_mse = ((imgs_res - imgs_gt) ** 2)[aux_pos_mask].mean().item() if aux_pos_mask.any() else float("nan")

                    raw_min = float(map_logits[0, 0].min().item())
                    raw_max = float(map_logits[0, 0].max().item())
                    mean_raw = float(map_logits[0, 0].mean().item())
                    print(f"[ep {ep} step {global_step}] loss={loss.item():.6f} bev={bev_loss.item():.6f} img={per_view_loss.item():.6f} "
                          f"pos_mse={pos_mse:.6f} aux_pos_mse={aux_pos_mse:.6f} "
                          f"pred_raw=[{raw_min:.3f},{raw_max:.3f}] mean={mean_raw:.3f} max_gt={float(map_gt[0,0].max().item()):.3f} lr={lr:.5f}")

            if global_step in save_steps:
                with torch.no_grad():
                    if fixed_item is not None:
                        fstem, fx, fmap_gt, _ = fixed_item
                        fx = fx.unsqueeze(0).to(dev)  # (1,V,3,Hi,Wi)
                        flogits, _ = model(fx)
                        fprob = torch.sigmoid(flogits[0, 0]).detach().cpu().numpy()

                        # pool gt to reduced size for fair visualization
                        fgt_pool = F.adaptive_max_pool2d(fmap_gt.unsqueeze(0), output_size=(Hb, Wb))[0, 0].cpu().numpy()

                        save_heat_png(out_dir / f"{fstem}_pred_step{global_step}.png", fprob)
                        save_heat_png(out_dir / f"{fstem}_gt_step{global_step}.png", fgt_pool)
                    else:
                        p0 = map_res[0, 0].detach().cpu().numpy()
                        g0 = F.adaptive_max_pool2d(map_gt, output_size=(Hb, Wb))[0, 0].detach().cpu().numpy()
                        save_heat_png(out_dir / f"{stems[0]}_pred_step{global_step}.png", p0)
                        save_heat_png(out_dir / f"{stems[0]}_gt_step{global_step}.png", g0)

            global_step += 1

    torch.save(model.state_dict(), out_dir / "model_multicam_mvdet_style_v3.pth")
    print("[OK] saved:", out_dir / "model_multicam_mvdet_style_v3.pth")


if __name__ == "__main__":
    main()
