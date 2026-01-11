# scripts/08_train_multicam_mvdet_style_v2.py
# 目标：更贴近 MVDet 论文定义的多视角 POM 训练（Wildtrack）
# - feature perspective transform + concat
# - ground-plane L2(Euclidean) loss on soft Gaussian target (sum/valid_norm, not mean)
# - per-view head/foot auxiliary heatmap regression loss (alpha=1 by default)
# - SGD(momentum=0.5, wd=5e-4) + OneCycleLR
# - backbone: resnet50 with replace_stride_with_dilation (stride=8 features) to mimic MVDet capacity
# - robust projection using cv2.projectPoints with distortion
# - auto drop bad views (valid_ratio too small)

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


CAM_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
IMG_ORI_W, IMG_ORI_H = 1920, 1080  # wildtrack original


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


def parse_rectangles_pom(pom_path: Path) -> Dict[str, float]:
    """
    尽量从 rectangles.pom 解析出 ORIGINE_X/ORIGINE_Y/STEP/NB_WIDTH/NB_HEIGHT。
    如果解析失败，返回空 dict，外部用 fallback。
    """
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
    # 常见 key：ORIGINE_X ORIGINE_Y STEP NB_WIDTH NB_HEIGHT
    return kv


def build_gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    xs = np.arange(size, dtype=np.float32) - radius
    ys = np.arange(size, dtype=np.float32) - radius
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    g /= (g.max() + 1e-12)
    return g.astype(np.float32)


def add_gaussian(heat: np.ndarray, cx: int, cy: int, kernel: np.ndarray):
    H, W = heat.shape
    kh, kw = kernel.shape
    r = kh // 2

    x0 = max(0, cx - r)
    x1 = min(H, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(W, cy + r + 1)

    kx0 = x0 - (cx - r)
    kx1 = kx0 + (x1 - x0)
    ky0 = y0 - (cy - r)
    ky1 = ky0 + (y1 - y0)

    if x0 >= x1 or y0 >= y1:
        return
    heat[x0:x1, y0:y1] = np.maximum(heat[x0:x1, y0:y1], kernel[kx0:kx1, ky0:ky1])


def save_heat_png(path: Path, arr: np.ndarray):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


class WildtrackMultiCam(Dataset):
    """
    - 输入：同一 stem 的多视角图像
    - ground GT：positionID -> BEV gaussian heatmap
    - aux GT：每个视角 bbox -> head/foot gaussian heatmap（在 feature 尺度上）
    """
    def __init__(
        self,
        data_root: Path,
        views: List[int],
        max_frames: int,
        img_hw: Tuple[int, int],   # (H,W) after resize
        feat_hw: Tuple[int, int],  # (Hf,Wf)
        bev_hw: Tuple[int, int],   # (Hb,Wb)
        bev_down: int,
        sigma_bev: float,
        sigma_img: float,
    ):
        self.data_root = data_root
        self.views = views
        self.Hi, self.Wi = img_hw
        self.Hf, self.Wf = feat_hw
        self.Hb, self.Wb = bev_hw
        self.bev_down = bev_down

        self.ann_dir = data_root / "annotations_positions"
        assert self.ann_dir.exists(), self.ann_dir

        self.img_dirs = []
        for v in views:
            p = data_root / "Image_subsets" / f"C{v+1}"
            assert p.exists(), p
            self.img_dirs.append(p)

        self.ann_files = sorted(self.ann_dir.glob("*.json"))
        if max_frames > 0:
            self.ann_files = self.ann_files[:max_frames]
        assert len(self.ann_files) > 0

        # kernels
        r_bev = int(np.ceil(3 * sigma_bev))
        self.k_bev = build_gaussian_kernel(sigma=sigma_bev, radius=r_bev)

        r_img = int(np.ceil(3 * sigma_img))
        self.k_img = build_gaussian_kernel(sigma=sigma_img, radius=r_img)

        # ImageNet normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # constants for pos_id mapping
        # NB_WIDTH=480 NB_HEIGHT=1440 in rectangles.pom
        self.NB_WIDTH = 480
        self.NB_HEIGHT = 1440

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

        # load multi-view images
        xs = []
        for img_dir in self.img_dirs:
            img = self._load_image(img_dir, stem).resize((self.Wi, self.Hi), Image.BILINEAR)
            x = torch.from_numpy(np.array(img, dtype=np.uint8)).float() / 255.0
            x = x.permute(2, 0, 1)
            x = (x - self.mean) / self.std
            xs.append(x)
        x = torch.stack(xs, dim=0)  # (V,3,Hi,Wi)

        # ground GT
        gt_bev = np.zeros((self.Hb, self.Wb), dtype=np.float32)

        # aux GT per view: (V,2,Hf,Wf) -> head,foot heatmaps
        gt_aux = np.zeros((len(self.views), 2, self.Hf, self.Wf), dtype=np.float32)

        for obj in data:
            pos_id = obj.get("positionID", None)
            if pos_id is None:
                continue
            pos_id = int(pos_id)
            ix = pos_id % self.NB_WIDTH
            iy = pos_id // self.NB_WIDTH
            gx = ix // self.bev_down
            gy = iy // self.bev_down
            if 0 <= gx < self.Hb and 0 <= gy < self.Wb:
                add_gaussian(gt_bev, gx, gy, self.k_bev)

            # aux: bbox per view
            views_dict = obj.get("views", {})
            # 兼容两种可能：key 是相机名或 "0"/"1"/...
            for vi, v in enumerate(self.views):
                cam_name = CAM_NAMES[v]
                bb = None
                if isinstance(views_dict, dict):
                    bb = views_dict.get(cam_name, None)
                    if bb is None:
                        bb = views_dict.get(str(v), None)
                # bbox 期望 [x1,y1,x2,y2] in original 1920x1080 coordinates
                if bb is None:
                    continue
                if not (isinstance(bb, (list, tuple)) and len(bb) >= 4):
                    continue
                x1, y1, x2, y2 = map(float, bb[:4])

                # scale bbox to resized image
                sx = self.Wi / IMG_ORI_W
                sy = self.Hi / IMG_ORI_H
                x1 *= sx; x2 *= sx
                y1 *= sy; y2 *= sy

                cx = 0.5 * (x1 + x2)
                head = (cx, y1)
                foot = (cx, y2)

                # map to feature coords (Hf,Wf) from image coords
                fx = self.Wf / self.Wi
                fy = self.Hf / self.Hi
                hx = int(np.clip(head[0] * fx, 0, self.Wf - 1))
                hy = int(np.clip(head[1] * fy, 0, self.Hf - 1))
                fxp = int(np.clip(foot[0] * fx, 0, self.Wf - 1))
                fyp = int(np.clip(foot[1] * fy, 0, self.Hf - 1))

                add_gaussian(gt_aux[vi, 0], hy, hx, self.k_img)  # 注意 heat 是 (H,W)，这里用 (y,x)
                add_gaussian(gt_aux[vi, 1], fyp, fxp, self.k_img)

        gt_bev_t = torch.from_numpy(gt_bev).unsqueeze(0)     # (1,Hb,Wb)
        gt_aux_t = torch.from_numpy(gt_aux)                  # (V,2,Hf,Wf)

        return stem, x, gt_bev_t, gt_aux_t


@torch.no_grad()
def build_bev_sampling_grid_projectpoints(
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    img_hw: Tuple[int, int],    # (Hi,Wi)
    feat_hw: Tuple[int, int],   # (Hf,Wf)
    bev_hw: Tuple[int, int],    # (Hb,Wb)
    origin_xy: Tuple[float, float],
    step: float,
    device: torch.device,
):
    """
    用 cv2.projectPoints(带畸变)做 BEV cell center -> image -> feature 的投影。
    """
    Hi, Wi = img_hw
    Hf, Wf = feat_hw
    Hb, Wb = bev_hw
    ox, oy = origin_xy

    # cell centers
    xs = ox + (np.arange(Hb, dtype=np.float64) + 0.5) * step
    ys = oy + (np.arange(Wb, dtype=np.float64) + 0.5) * step
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (Hb,Wb)

    objp = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3).astype(np.float64)

    # project
    img_pts, _ = cv2.projectPoints(objp, rvec.astype(np.float64), tvec.astype(np.float64), K.astype(np.float64), dist.astype(np.float64))
    img_pts = img_pts.reshape(-1, 2)
    u = img_pts[:, 0]
    v = img_pts[:, 1]

    valid = (u >= 0) & (u <= (Wi - 1)) & (v >= 0) & (v <= (Hi - 1))

    # map to feature coords (aligned to endpoints)
    uf = u * (Wf - 1) / (Wi - 1)
    vf = v * (Hf - 1) / (Hi - 1)

    x_norm = 2.0 * (uf / (Wf - 1)) - 1.0
    y_norm = 2.0 * (vf / (Hf - 1)) - 1.0

    grid = np.stack([x_norm, y_norm], axis=-1).reshape(Hb, Wb, 2).astype(np.float32)
    valid_m = valid.reshape(Hb, Wb).astype(np.float32)

    grid_t = torch.from_numpy(grid).unsqueeze(0).to(device)                    # (1,Hb,Wb,2)
    valid_t = torch.from_numpy(valid_m).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,Hb,Wb)

    # invalid -> out of range => grid_sample=0
    grid_t = torch.where(valid_t.permute(0, 2, 3, 1) > 0, grid_t, torch.full_like(grid_t, 2.0))
    return grid_t, valid_t


class ResNet50DilatedStride8(nn.Module):
    """
    torchvision resnet50 bottleneck 支持 dilation。
    replace_stride_with_dilation=[False, True, True] -> 最后两层不下采样，用 dilation，输出 stride=8。
    """
    def __init__(self, pretrained: bool, out_ch: int = 512):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4  # stride=8 with dilation
        self.reduce = nn.Conv2d(2048, out_ch, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.reduce(x)
        return x  # (B,out_ch,H/8,W/8)


class GNRelu(nn.Module):
    def __init__(self, ch: int, groups: int = 32):
        super().__init__()
        g = min(groups, ch)
        self.gn = nn.GroupNorm(g, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(x))


class BEVHeadDilated(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 256):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, dilation=1)
        self.n1 = GNRelu(mid_ch)
        self.c2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=2, dilation=2)
        self.n2 = GNRelu(mid_ch)
        self.c3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=4, dilation=4)
        self.n3 = GNRelu(mid_ch)
        self.out = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x):
        x = self.n1(self.c1(x))
        x = self.n2(self.c2(x))
        x = self.n3(self.c3(x))
        x = self.out(x)
        return x  # logits


class AuxHeadFoot(nn.Module):
    """
    在每视角 feature map 上预测 head/foot 2通道 heatmap（logits）。
    """
    def __init__(self, in_ch: int, mid_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            GNRelu(mid_ch),
            nn.Conv2d(mid_ch, 2, 1),
        )

    def forward(self, f):
        return self.net(f)  # (B,2,Hf,Wf) logits


class MVDetLikeMultiCam(nn.Module):
    def __init__(
        self,
        num_views: int,
        bev_hw: Tuple[int, int],
        grids: torch.Tensor,   # (V,1,Hb,Wb,2)
        valids: torch.Tensor,  # (V,1,1,Hb,Wb)
        feat_hw: Tuple[int, int],
        pretrained: bool,
        feat_ch: int = 512,
    ):
        super().__init__()
        self.num_views = num_views
        self.Hb, self.Wb = bev_hw
        self.Hf, self.Wf = feat_hw

        self.backbone = ResNet50DilatedStride8(pretrained=pretrained, out_ch=feat_ch)
        self.aux = AuxHeadFoot(in_ch=feat_ch, mid_ch=128)

        self.grids = nn.Parameter(grids, requires_grad=False)
        self.valids = nn.Parameter(valids, requires_grad=False)

        # coord map
        xs = torch.linspace(-1, 1, self.Hb).view(self.Hb, 1).expand(self.Hb, self.Wb)
        ys = torch.linspace(-1, 1, self.Wb).view(1, self.Wb).expand(self.Hb, self.Wb)
        coord = torch.stack([xs, ys], dim=0).unsqueeze(0)  # (1,2,Hb,Wb)
        self.coord = nn.Parameter(coord, requires_grad=False)

        in_bev = num_views * feat_ch + 2
        self.head = BEVHeadDilated(in_ch=in_bev, mid_ch=256)

    def forward(self, x_views: torch.Tensor):
        """
        x_views: (B,V,3,Hi,Wi)
        return:
          bev_logits: (B,1,Hb,Wb)
          aux_logits: (B,V,2,Hf,Wf)
        """
        B, V, _, _, _ = x_views.shape
        feats = []
        auxs = []
        for vi in range(V):
            f = self.backbone(x_views[:, vi])  # (B,C,Hi/8,Wi/8)
            f = F.interpolate(f, size=(self.Hf, self.Wf), mode="bilinear", align_corners=False)
            aux_logits = self.aux(f)  # (B,2,Hf,Wf)
            auxs.append(aux_logits)

            grid = self.grids[vi].expand(B, -1, -1, -1)            # (B,Hb,Wb,2)
            valid = self.valids[vi].expand(B, -1, -1, -1)          # (B,1,Hb,Wb)
            bev = F.grid_sample(f, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            bev = bev * valid
            feats.append(bev)

        bev_cat = torch.cat(feats, dim=1)  # (B,V*C,Hb,Wb)
        coord = self.coord.to(bev_cat.device).expand(B, -1, -1, -1)
        bev_cat = torch.cat([bev_cat, coord], dim=1)

        bev_logits = self.head(bev_cat)
        aux_logits = torch.stack(auxs, dim=1)  # (B,V,2,Hf,Wf)
        return bev_logits, aux_logits


def l2_map_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, reduction: str = "valid_mean"):
    """
    pred/target: (B,1,H,W) or (B,C,H,W)
    mask: (B,1,H,W) broadcastable
    reduction:
      - "sum": sum of squared error
      - "mean": mean over all elements
      - "valid_mean": sum / (mask.sum)
    """
    diff2 = (pred - target) ** 2
    if mask is not None:
        diff2 = diff2 * mask
        if reduction == "valid_mean":
            denom = mask.sum().clamp_min(1.0)
            return diff2.sum() / denom
    if reduction == "sum":
        return diff2.sum()
    return diff2.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="wildtrack")
    ap.add_argument("--views", type=str, default="0,1,2", help="e.g. 0,3,5 or 'auto:0,1,2,3,4,5,6'")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--bev_down", type=int, default=4)

    ap.add_argument("--sigma_bev", type=float, default=2.5)
    ap.add_argument("--sigma_img", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="aux loss weight (MVDet uses alpha=1)")
    ap.add_argument("--pretrained", action="store_true")

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_lr", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--freeze_bn", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_steps", type=str, default="0,200,1000")
    ap.add_argument("--fixed_stem", type=str, default="")

    ap.add_argument("--drop_bad_views", action="store_true", help="auto drop views with very small valid_ratio")
    ap.add_argument("--valid_thr", type=float, default=0.1)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path("outputs") / "train_multicam_mvdet_style_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device)
    print("[DEV] device =", dev, "cuda_available=", torch.cuda.is_available())
    if dev.type == "cuda":
        print("[DEV] gpu =", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # MVDet settings from paper: input resized and feature resized
    Hi, Wi = 720, 1280
    Hf, Wf = 270, 480

    # rectangles.pom parse (fallback to your known values)
    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    NB_WIDTH = int(pom.get("NB_WIDTH", 480))
    NB_HEIGHT = int(pom.get("NB_HEIGHT", 1440))
    ORIGINE_X = float(pom.get("ORIGINE_X", -3.0))
    ORIGINE_Y = float(pom.get("ORIGINE_Y", -9.0))
    STEP = float(pom.get("STEP", 0.025))

    Hb = NB_WIDTH // args.bev_down
    Wb = NB_HEIGHT // args.bev_down
    print(f"[CFG] img={Hi}x{Wi}, feat={Hf}x{Wf}, bev={Hb}x{Wb}, bev_down={args.bev_down}")
    print(f"[POM] NB_WIDTH={NB_WIDTH} NB_HEIGHT={NB_HEIGHT} ORIGINE=({ORIGINE_X},{ORIGINE_Y}) STEP={STEP}")

    # parse views
    views_str = args.views.strip()
    auto_mode = views_str.startswith("auto:")
    if auto_mode:
        views = [int(x) for x in views_str.replace("auto:", "").split(",") if x.strip().isdigit()]
    else:
        views = [int(x) for x in views_str.split(",") if x.strip().isdigit()]
    assert len(views) > 0

    # calib and grids
    calib_root = data_root / "calibrations"
    grids = []
    valids = []
    kept_views = []
    for v in views:
        cam_name = CAM_NAMES[v]
        intr_path = calib_root / "intrinsic_original" / f"intr_{cam_name}.xml"
        extr_path = calib_root / "extrinsic" / f"extr_{cam_name}.xml"
        assert intr_path.exists(), intr_path
        assert extr_path.exists(), extr_path

        K0, dist = read_intrinsics(intr_path)
        sx = Wi / IMG_ORI_W
        sy = Hi / IMG_ORI_H
        K = scale_intrinsics(K0, sx=sx, sy=sy)

        rvec = read_opencv_xml_vec(extr_path, "rvec").reshape(3, 1)
        tvec = read_opencv_xml_vec(extr_path, "tvec").reshape(3, 1)

        # 经验性单位提示：如果 tvec 范数很大（>50），很可能是 cm；而你的 ORIGINE/STEP 如果是 m（0.025），就会混单位
        t_norm = float(np.linalg.norm(tvec))
        if STEP < 1.0 and t_norm > 50.0:
            # world in meters but t in cm -> convert t to meters
            tvec = tvec / 100.0

        grid, valid = build_bev_sampling_grid_projectpoints(
            K=K, dist=dist, rvec=rvec, tvec=tvec,
            img_hw=(Hi, Wi), feat_hw=(Hf, Wf), bev_hw=(Hb, Wb),
            origin_xy=(ORIGINE_X, ORIGINE_Y),
            step=STEP * args.bev_down,
            device=dev,
        )
        vr = float(valid.mean().item())
        print(f"[GRID] view={v} cam={cam_name} valid_ratio={vr:.4f}")

        if args.drop_bad_views and vr < args.valid_thr:
            print(f"[GRID] drop view={v} due to valid_ratio<{args.valid_thr}")
            continue

        grids.append(grid)
        valids.append(valid)
        kept_views.append(v)

    assert len(kept_views) > 0, "No valid views left. Fix calibration / unit mismatch first."
    print("[CFG] kept_views =", kept_views)

    grids_t = torch.cat(grids, dim=0)     # (V,1,Hb,Wb,2)
    valids_t = torch.cat(valids, dim=0)   # (V,1,1,Hb,Wb)

    ds = WildtrackMultiCam(
        data_root=data_root,
        views=kept_views,
        max_frames=args.max_frames,
        img_hw=(Hi, Wi),
        feat_hw=(Hf, Wf),
        bev_hw=(Hb, Wb),
        bev_down=args.bev_down,
        sigma_bev=args.sigma_bev,
        sigma_img=args.sigma_img,
    )

    def collate(batch):
        stems, xs, gt_bev, gt_aux = zip(*batch)
        return list(stems), torch.stack(xs, 0), torch.stack(gt_bev, 0), torch.stack(gt_aux, 0)

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

    model = MVDetLikeMultiCam(
        num_views=len(kept_views),
        bev_hw=(Hb, Wb),
        grids=grids_t,
        valids=valids_t,
        feat_hw=(Hf, Wf),
        pretrained=args.pretrained,
        feat_ch=512,
    ).to(dev)

    # freeze BN if needed (batch=1)
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        print("[CFG] freeze_bn = True")

    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=5e-4)
    steps_per_epoch = len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    save_steps = set(int(s.strip()) for s in args.save_steps.split(",") if s.strip().isdigit())

    # fixed stem
    fixed_item = None
    if args.fixed_stem:
        ann = data_root / "annotations_positions" / f"{args.fixed_stem}.json"
        if ann.exists():
            for i in range(len(ds)):
                stem, x, gt_bev, gt_aux = ds[i]
                if stem == args.fixed_stem:
                    fixed_item = (stem, x, gt_bev, gt_aux)
                    print("[FIXED] using stem =", args.fixed_stem)
                    break

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and dev.type == "cuda"))

    global_step = 0
    model.train()

    for ep in range(args.epochs):
        for stems, x_views, gt_bev, gt_aux in loader:
            x_views = x_views.to(dev, non_blocking=True)  # (B,V,3,Hi,Wi)
            gt_bev = gt_bev.to(dev, non_blocking=True)    # (B,1,Hb,Wb)
            gt_aux = gt_aux.to(dev, non_blocking=True)    # (B,V,2,Hf,Wf)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                bev_logits, aux_logits = model(x_views)

                bev_prob = torch.sigmoid(bev_logits)
                aux_prob = torch.sigmoid(aux_logits)

                # union valid mask for ground loss
                # valids_t: (V,1,1,Hb,Wb) -> union (1,1,Hb,Wb)
                union_valid = (model.valids.sum(dim=0) > 0).float()  # (1,1,Hb,Wb)
                union_valid = union_valid.expand(bev_prob.shape[0], -1, -1, -1)

                loss_ground = l2_map_loss(bev_prob, gt_bev, mask=union_valid, reduction="valid_mean")

                # aux loss: only on feature maps; average over views
                # mask not needed; head/foot targets are already sparse but at least provide view-specific gradients
                loss_aux = ((aux_prob - gt_aux) ** 2).mean()

                loss = loss_ground + args.alpha * loss_aux

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # IMPORTANT: scheduler after optimizer.step()
            sched.step()

            if global_step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                with torch.no_grad():
                    pos_mask = (gt_bev > 0.1) & (union_valid > 0)
                    pos_mse = ((bev_prob - gt_bev) ** 2)[pos_mask].mean().item() if pos_mask.any() else float("nan")
                    aux_pos_mask = (gt_aux > 0.1)
                    aux_pos_mse = ((aux_prob - gt_aux) ** 2)[aux_pos_mask].mean().item() if aux_pos_mask.any() else float("nan")

                    raw_min = float(bev_logits[0, 0].min().item())
                    raw_max = float(bev_logits[0, 0].max().item())
                    mean_raw = float(bev_logits[0, 0].mean().item())
                    print(f"[ep {ep} step {global_step}] loss={loss.item():.6f} ground={loss_ground.item():.6f} aux={loss_aux.item():.6f} "
                          f"pos_mse={pos_mse:.6f} aux_pos_mse={aux_pos_mse:.6f} "
                          f"pred_bev_raw=[{raw_min:.3f},{raw_max:.3f}] mean={mean_raw:.3f} max_gt={gt_bev[0,0].max().item():.3f} lr={lr:.5f}")

            if global_step in save_steps:
                with torch.no_grad():
                    if fixed_item is not None:
                        fstem, fx, fgt_bev, fgt_aux = fixed_item
                        fx = fx.unsqueeze(0).to(dev)  # (1,V,3,Hi,Wi)
                        f_bev_logits, _ = model(fx)
                        f_bev_prob = torch.sigmoid(f_bev_logits[0, 0]).detach().cpu().numpy()
                        f_gt = fgt_bev[0].detach().cpu().numpy()
                        save_heat_png(out_dir / f"{fstem}_pred_step{global_step}.png", f_bev_prob)
                        save_heat_png(out_dir / f"{fstem}_gt_step{global_step}.png", f_gt)
                    else:
                        p0 = torch.sigmoid(bev_logits[0, 0]).detach().cpu().numpy()
                        g0 = gt_bev[0, 0].detach().cpu().numpy()
                        save_heat_png(out_dir / f"{stems[0]}_pred_step{global_step}.png", p0)
                        save_heat_png(out_dir / f"{stems[0]}_gt_step{global_step}.png", g0)

            global_step += 1

    torch.save(model.state_dict(), out_dir / "model_multicam_mvdet_style_v2.pth")
    print("[OK] saved:", out_dir / "model_multicam_mvdet_style_v2.pth")


if __name__ == "__main__":
    main()
