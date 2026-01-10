# scripts/08_train_multicam_mvdet_style.py
# Multi-view MVDet-style training on Wildtrack:
# - Shared ResNet18 (stride=8 with dilation) backbone
# - Per-view feature -> BEV warp (grid_sample) -> per-view 1x1 reduce -> concat fusion
# - BEV head: dilated conv (1/2/4) -> 1ch ground-plane POM (regression)
# - Loss:
#   (1) ground-plane soft Gaussian target + MSE
#   (2) per-view head/foot auxiliary heatmap regression (from bboxes) + MSE
# - View-coherent photometric augmentation (same params applied to all views for a frame)
# - AMP (mixed precision) for MX450 feasibility

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision


# ---- Wildtrack rectangles.pom constants ----
ORIGINE_X_M = -3.0
ORIGINE_Y_M = -9.0
NB_WIDTH = 480
NB_HEIGHT = 1440
WIDTH_M = 12.0
STEP_M = WIDTH_M / NB_WIDTH  # 0.025m
CAM_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]


# ------------------ Calibration utils ------------------
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


# ------------------ Heatmap utils ------------------
def build_gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    xs = np.arange(size, dtype=np.float32) - radius
    ys = np.arange(size, dtype=np.float32) - radius
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    g /= (g.max() + 1e-12)
    return g.astype(np.float32)


def add_gaussian_max(heat: np.ndarray, cx: int, cy: int, kernel: np.ndarray):
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


# ------------------ BEV sampling grid ------------------
@torch.no_grad()
def build_bev_sampling_grid(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    img_hw: Tuple[int, int],   # (H,W)
    feat_hw: Tuple[int, int],  # (Hf,Wf)
    bev_hw: Tuple[int, int],   # (Hg,Wg)
    bev_down: int,
    device: torch.device,
):
    Hi, Wi = img_hw
    Hf, Wf = feat_hw
    Hg, Wg = bev_hw

    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    t = tvec.astype(np.float64).reshape(3, 1)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    step = STEP_M * bev_down
    ox = ORIGINE_X_M + 0.5 * step
    oy = ORIGINE_Y_M + 0.5 * step

    xs = ox + np.arange(Hg, dtype=np.float64) * step
    ys = oy + np.arange(Wg, dtype=np.float64) * step
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (Hg,Wg)

    Xw = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3).T  # (3,N)
    Xc = (R @ Xw) + t
    z = Xc[2, :]
    valid = z > 1e-6

    u = fx * (Xc[0, :] / z) + cx
    v = fy * (Xc[1, :] / z) + cy

    valid = valid & (u >= 0) & (u <= (Wi - 1)) & (v >= 0) & (v <= (Hi - 1))

    # image -> feature coords
    uf = u * (Wf - 1) / (Wi - 1)
    vf = v * (Hf - 1) / (Hi - 1)

    x_norm = 2.0 * (uf / (Wf - 1)) - 1.0
    y_norm = 2.0 * (vf / (Hf - 1)) - 1.0

    grid = np.stack([x_norm, y_norm], axis=-1).reshape(Hg, Wg, 2).astype(np.float32)
    valid_m = valid.reshape(Hg, Wg).astype(np.float32)

    grid_t = torch.from_numpy(grid).unsqueeze(0).to(device)                    # (1,Hg,Wg,2)
    valid_t = torch.from_numpy(valid_m).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,Hg,Wg)

    # invalid -> out of range => grid_sample gives 0
    grid_t = torch.where(valid_t.permute(0, 2, 3, 1) > 0, grid_t, torch.full_like(grid_t, 2.0))
    return grid_t, valid_t


# ------------------ ResNet18 (stride=8 with dilation support) ------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, dilation=dilation, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockDilated(nn.Module):
    expansion = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18Stride8Dilated(nn.Module):
    """
    Output stride=8, channels=512.
    replace_stride_with_dilation for layer3/layer4 -> dilation 2/4.
    """
    def __init__(self, replace_stride_with_dilation=(False, True, True), norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2, stride=1, dilate=False)
        self.layer2 = self._make_layer(128, blocks=2, stride=2, dilate=False)  # stride=8 after this
        self.layer3 = self._make_layer(256, blocks=2, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, blocks=2, stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int, dilate: bool):
        norm_layer = self.norm_layer
        downsample = None
        prev_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        dilation = self.dilation

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(BasicBlockDilated(self.inplanes, planes, stride=stride, downsample=downsample,
                                        dilation=prev_dilation if not dilate else dilation, norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockDilated(self.inplanes, planes, stride=1, downsample=None,
                                            dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B,512,Hi/8,Wi/8)


def build_resnet18_stride8_dilated(pretrained: bool) -> ResNet18Stride8Dilated:
    m = ResNet18Stride8Dilated(replace_stride_with_dilation=(False, True, True))
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        tv = torchvision.models.resnet18(weights=weights)
        sd = tv.state_dict()
        m.load_state_dict(sd, strict=False)
    return m


# ------------------ Heads ------------------
class BEVHeadDilated(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, mid_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, mid_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, 1, 1, bias=True),  # regression output
        )

    def forward(self, x):
        return self.net(x)  # (B,1,Hg,Wg)


class AuxHeadFootHead(nn.Module):
    """
    Per-view auxiliary head/foot heatmap predictor on stride=8 feature map.
    Output: (B,2,Hs,Ws) where ch0=foot, ch1=head
    """
    def __init__(self, in_ch: int = 512, mid_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 2, 1, bias=True),
        )

    def forward(self, f_s8):
        return self.net(f_s8)


def freeze_batchnorm(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False


# ------------------ Dataset ------------------
def parse_views_bboxes(obj: Dict[str, Any]) -> Optional[List[Optional[List[float]]]]:
    """
    Try to parse bboxes per view from the annotation object.
    We try to be robust:
    - views can be a list length 7: each entry either bbox list or None
    - or a dict keyed by cam index/name
    """
    v = obj.get("views", None)
    if v is None:
        return None
    if isinstance(v, list):
        # entries could be dicts or lists
        out: List[Optional[List[float]]] = []
        for item in v:
            if item is None:
                out.append(None)
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                out.append([float(item[0]), float(item[1]), float(item[2]), float(item[3])])
            elif isinstance(item, dict):
                # common keys: x,y,w,h or xmin,ymin,xmax,ymax
                if all(k in item for k in ["xmin", "ymin", "xmax", "ymax"]):
                    out.append([float(item["xmin"]), float(item["ymin"]), float(item["xmax"]), float(item["ymax"])])
                elif all(k in item for k in ["x", "y", "w", "h"]):
                    x, y, w, h = float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"])
                    out.append([x, y, x + w, y + h])
                else:
                    out.append(None)
            else:
                out.append(None)
        # pad to 7 if needed
        while len(out) < 7:
            out.append(None)
        return out[:7]

    if isinstance(v, dict):
        out = [None] * 7
        for k, item in v.items():
            idx = None
            if isinstance(k, int):
                idx = k
            elif isinstance(k, str):
                if k.isdigit():
                    idx = int(k)
                else:
                    # maybe cam name
                    if k in CAM_NAMES:
                        idx = CAM_NAMES.index(k)
            if idx is None or not (0 <= idx < 7):
                continue
            if item is None:
                out[idx] = None
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                out[idx] = [float(item[0]), float(item[1]), float(item[2]), float(item[3])]
            elif isinstance(item, dict):
                if all(kk in item for kk in ["xmin", "ymin", "xmax", "ymax"]):
                    out[idx] = [float(item["xmin"]), float(item["ymin"]), float(item["xmax"]), float(item["ymax"])]
                elif all(kk in item for kk in ["x", "y", "w", "h"]):
                    x, y, w, h = float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"])
                    out[idx] = [x, y, x + w, y + h]
        return out
    return None


class WildtrackMultiCamPOM(Dataset):
    """
    Multi-view images + ground-plane POM target + per-view aux (head/foot heatmaps from bboxes).
    """
    def __init__(
        self,
        data_root: Path,
        views: List[int],
        max_frames: int,
        bev_down: int,
        sigma_bev: float,
        sigma_aux: float,
        img_size: Tuple[int, int],  # (Hi,Wi)
        photometric_aug: bool,
        seed: int = 0,
    ):
        self.data_root = data_root
        self.views = views
        self.bev_down = bev_down
        self.sigma_bev = sigma_bev
        self.sigma_aux = sigma_aux
        self.photometric_aug = photometric_aug
        self.rng = np.random.RandomState(seed)

        self.Hi, self.Wi = img_size
        self.Hg = NB_WIDTH // bev_down
        self.Wg = NB_HEIGHT // bev_down

        # aux heatmap on stride=8 feature size
        self.Hs = self.Hi // 8
        self.Ws = self.Wi // 8

        self.ann_dir = data_root / "annotations_positions"
        assert self.ann_dir.exists(), f"missing {self.ann_dir}"
        self.ann_files = sorted(self.ann_dir.glob("*.json"))
        if max_frames > 0:
            self.ann_files = self.ann_files[:max_frames]
        assert len(self.ann_files) > 0

        # image dirs per selected view
        self.img_dirs = []
        for v in self.views:
            d = data_root / "Image_subsets" / f"C{v+1}"
            assert d.exists(), f"missing {d}"
            self.img_dirs.append(d)

        # kernels
        r_bev = int(np.ceil(3 * sigma_bev))
        self.k_bev = build_gaussian_kernel(sigma_bev, r_bev)
        r_aux = int(np.ceil(3 * sigma_aux))
        self.k_aux = build_gaussian_kernel(sigma_aux, r_aux)

        # ImageNet normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # scale from original wildtrack images (1080x1920) to (Hi,Wi)
        self.sx = self.Wi / 1920.0
        self.sy = self.Hi / 1080.0

    def __len__(self):
        return len(self.ann_files)

    def _load_image_from_dir(self, img_dir: Path, stem: str) -> Image.Image:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"missing image for {stem} in {img_dir}")

    def _view_coherent_photometric(self, imgs: List[Image.Image]) -> List[Image.Image]:
        """
        Apply the same photometric transform parameters to all views in a frame.
        We avoid geometric aug to keep calibration valid.
        """
        if not self.photometric_aug:
            return imgs

        # sample parameters
        # brightness/contrast/saturation in [0.8,1.2], gamma in [0.8,1.2]
        b = float(self.rng.uniform(0.8, 1.2))
        c = float(self.rng.uniform(0.8, 1.2))
        s = float(self.rng.uniform(0.8, 1.2))
        g = float(self.rng.uniform(0.8, 1.2))
        noise_std = float(self.rng.uniform(0.0, 0.02))

        out = []
        for im in imgs:
            x = np.asarray(im).astype(np.float32) / 255.0  # H,W,3
            # brightness
            x = x * b
            # contrast around mean
            mu = x.mean(axis=(0, 1), keepdims=True)
            x = (x - mu) * c + mu
            # saturation (approx in RGB space)
            gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
            x = (x - gray) * s + gray
            # gamma
            x = np.clip(x, 0, 1)
            x = x ** g
            # noise
            if noise_std > 0:
                x = x + self.rng.normal(0, noise_std, size=x.shape).astype(np.float32)
            x = np.clip(x, 0, 1)
            out.append(Image.fromarray((x * 255.0).astype(np.uint8)))
        return out

    def __getitem__(self, idx: int):
        ann_path = self.ann_files[idx]
        stem = ann_path.stem
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        # Load multi-view images
        imgs = [self._load_image_from_dir(d, stem) for d in self.img_dirs]
        imgs = [im.resize((self.Wi, self.Hi), Image.BILINEAR) for im in imgs]
        imgs = self._view_coherent_photometric(imgs)

        # to tensor + normalize
        xs = []
        for im in imgs:
            x = torch.from_numpy(np.array(im, dtype=np.uint8)).float() / 255.0  # (H,W,3)
            x = x.permute(2, 0, 1)  # (3,H,W)
            x = (x - self.mean) / self.std
            xs.append(x)
        x_mv = torch.stack(xs, dim=0)  # (V,3,Hi,Wi)

        # Ground-plane GT (BEV)
        gt_bev = np.zeros((self.Hg, self.Wg), dtype=np.float32)

        # Aux GT per selected view: (V,2,Hs,Ws)
        gt_aux = np.zeros((len(self.views), 2, self.Hs, self.Ws), dtype=np.float32)

        for obj in data:
            pos_id = obj.get("positionID", None)
            if pos_id is not None:
                pos_id = int(pos_id)
                ix = pos_id % NB_WIDTH
                iy = pos_id // NB_WIDTH
                gx = ix // self.bev_down
                gy = iy // self.bev_down
                if 0 <= gx < self.Hg and 0 <= gy < self.Wg:
                    add_gaussian_max(gt_bev, gx, gy, self.k_bev)

            # per-view head/foot from bbox
            bboxes_all = parse_views_bboxes(obj)
            if bboxes_all is None:
                continue
            for vi, v in enumerate(self.views):
                bb = bboxes_all[v] if v < len(bboxes_all) else None
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                # scale to resized image (Hi,Wi) from original (1080,1920)
                x1 *= self.sx; x2 *= self.sx
                y1 *= self.sy; y2 *= self.sy

                # head/foot points in resized image coords
                u = 0.5 * (x1 + x2)
                v_head = y1
                v_foot = y2

                # map to stride=8 feature coords (Hs,Ws)
                uf = u * (self.Ws - 1) / (self.Wi - 1)
                vh = v_head * (self.Hs - 1) / (self.Hi - 1)
                vf = v_foot * (self.Hs - 1) / (self.Hi - 1)

                cx = int(round(vf))   # row index
                cy = int(round(uf))   # col index
                if 0 <= cx < self.Hs and 0 <= cy < self.Ws:
                    add_gaussian_max(gt_aux[vi, 0], cx, cy, self.k_aux)  # foot

                cx = int(round(vh))
                cy = int(round(uf))
                if 0 <= cx < self.Hs and 0 <= cy < self.Ws:
                    add_gaussian_max(gt_aux[vi, 1], cx, cy, self.k_aux)  # head

        gt_bev_t = torch.from_numpy(gt_bev).unsqueeze(0)  # (1,Hg,Wg)
        gt_aux_t = torch.from_numpy(gt_aux)               # (V,2,Hs,Ws)

        return stem, x_mv, gt_bev_t, gt_aux_t


# ------------------ Model ------------------
class MVDetMultiCam(nn.Module):
    def __init__(
        self,
        num_views: int,
        bev_hw: Tuple[int, int],
        grids: List[torch.Tensor],
        valids: List[torch.Tensor],
        reduce_ch: int = 64,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.backbone = build_resnet18_stride8_dilated(pretrained=pretrained)

        # register grids/valids as buffers
        for i in range(num_views):
            self.register_buffer(f"grid_{i}", grids[i], persistent=False)    # (1,Hg,Wg,2)
            self.register_buffer(f"valid_{i}", valids[i], persistent=False)  # (1,1,Hg,Wg)

        Hg, Wg = bev_hw
        xs = torch.linspace(-1, 1, Hg).view(Hg, 1).expand(Hg, Wg)
        ys = torch.linspace(-1, 1, Wg).view(1, Wg).expand(Hg, Wg)
        coord = torch.stack([xs, ys], dim=0).unsqueeze(0)  # (1,2,Hg,Wg)
        self.register_buffer("coord", coord, persistent=False)

        # per-view BEV channel reduce (512 -> reduce_ch)
        self.reduce = nn.Sequential(
            nn.Conv2d(512, reduce_ch, 1, bias=False),
            nn.BatchNorm2d(reduce_ch),
            nn.ReLU(inplace=True),
        )

        # ground-plane head: input = num_views*reduce_ch + 2
        self.bev_head = BEVHeadDilated(in_ch=num_views * reduce_ch + 2, mid_ch=256)

        # per-view aux head on stride=8 features
        self.aux_head = AuxHeadFootHead(in_ch=512, mid_ch=128)

    def forward(self, x_mv: torch.Tensor):
        """
        x_mv: (B,V,3,Hi,Wi)
        returns:
          pred_bev: (B,1,Hg,Wg)
          pred_aux: (B,V,2,Hs,Ws)
        """
        B, V, C, Hi, Wi = x_mv.shape
        assert V == self.num_views

        bev_feats = []
        aux_preds = []

        for i in range(V):
            x = x_mv[:, i]  # (B,3,Hi,Wi)
            f = self.backbone(x)  # (B,512,Hs,Ws) where Hs=Hi/8, Ws=Wi/8
            aux = self.aux_head(f)  # (B,2,Hs,Ws)
            aux_preds.append(aux)

            # interpolate to fixed feat size before projection (MVDet style)
            f_up = F.interpolate(f, size=(270, 480), mode="bilinear", align_corners=False)

            grid = getattr(self, f"grid_{i}").expand(B, -1, -1, -1)
            valid = getattr(self, f"valid_{i}").expand(B, -1, -1, -1)

            bev = F.grid_sample(f_up, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            bev = bev * valid

            bev = self.reduce(bev)  # (B,reduce_ch,Hg,Wg)
            bev_feats.append(bev)

        bev_cat = torch.cat(bev_feats, dim=1)  # (B,V*reduce_ch,Hg,Wg)
        coord = self.coord.expand(B, -1, -1, -1).to(bev_cat.device)
        bev_in = torch.cat([bev_cat, coord], dim=1)

        pred_bev = self.bev_head(bev_in)  # (B,1,Hg,Wg)
        pred_aux = torch.stack(aux_preds, dim=1)  # (B,V,2,Hs,Ws)
        return pred_bev, pred_aux


# ------------------ Main train ------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="wildtrack")
    parser.add_argument("--views", type=str, default="0,1,2", help="comma sep view indices, e.g. 0,1,2 or 0,1,2,3,4,5,6")
    parser.add_argument("--max_frames", type=int, default=300)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--bev_down", type=int, default=4)

    parser.add_argument("--sigma_bev", type=float, default=3.0)
    parser.add_argument("--sigma_aux", type=float, default=2.0)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--photometric_aug", action="store_true")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_lr", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--reduce_ch", type=int, default=64, help="per-view BEV channel reduction before concat")
    parser.add_argument("--aux_w", type=float, default=1.0, help="weight for per-view aux loss")
    parser.add_argument("--ground_w", type=float, default=1.0, help="weight for ground-plane loss")
    parser.add_argument("--loss_pos_w", type=float, default=1.0, help="optional pos-region weighting for ground loss")
    parser.add_argument("--pos_thr", type=float, default=0.1)

    parser.add_argument("--amp", action="store_true", help="enable torch.cuda.amp for speed/memory")
    parser.add_argument("--viz_sigmoid", action="store_true", help="only for saving pred images")
    parser.add_argument("--fixed_stem", type=str, default="")
    parser.add_argument("--save_steps", type=str, default="0,50,100,200,400,800")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    views = [int(s.strip()) for s in args.views.split(",") if s.strip() != ""]
    assert len(views) >= 1
    for v in views:
        assert 0 <= v < 7

    data_root = Path(args.data_root)
    out_dir = Path("outputs") / "train_multicam_mvdet_style"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda but torch.cuda.is_available() is False.")
    dev = torch.device(args.device)
    print("[DEV] device =", dev, "cuda_available=", torch.cuda.is_available())
    if dev.type == "cuda":
        print("[DEV] gpu =", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # fixed image/feature sizes (MVDet style)
    Hi, Wi = 720, 1280
    Hf, Wf = 270, 480

    Hg = NB_WIDTH // args.bev_down
    Wg = NB_HEIGHT // args.bev_down
    print(f"[CFG] views={views}, img={Hi}x{Wi}, feat={Hf}x{Wf}, bev={Hg}x{Wg}, bev_down={args.bev_down}")

    # Build grids for each selected view
    calib_root = data_root / "calibrations"
    grids: List[torch.Tensor] = []
    valids: List[torch.Tensor] = []
    for v in views:
        cam_name = CAM_NAMES[v]
        intr_path = calib_root / "intrinsic_original" / f"intr_{cam_name}.xml"
        extr_path = calib_root / "extrinsic" / f"extr_{cam_name}.xml"
        assert intr_path.exists(), intr_path
        assert extr_path.exists(), extr_path

        K0, _dist = read_intrinsics(intr_path)
        sx = Wi / 1920.0
        sy = Hi / 1080.0
        K = scale_intrinsics(K0, sx=sx, sy=sy)

        rvec = read_opencv_xml_vec(extr_path, "rvec").reshape(3, 1)
        tvec = read_opencv_xml_vec(extr_path, "tvec").reshape(3, 1)

        grid, valid = build_bev_sampling_grid(
            K=K, rvec=rvec, tvec=tvec,
            img_hw=(Hi, Wi), feat_hw=(Hf, Wf), bev_hw=(Hg, Wg),
            bev_down=args.bev_down, device=dev
        )
        grids.append(grid)
        valids.append(valid)
        print(f"[GRID] view={v} cam={cam_name} valid_ratio={valid.mean().item():.4f}")

    ds = WildtrackMultiCamPOM(
        data_root=data_root,
        views=views,
        max_frames=args.max_frames,
        bev_down=args.bev_down,
        sigma_bev=args.sigma_bev,
        sigma_aux=args.sigma_aux,
        img_size=(Hi, Wi),
        photometric_aug=args.photometric_aug,
        seed=args.seed,
    )

    def collate(batch):
        stems, x_mv, gt_bev, gt_aux = zip(*batch)
        return list(stems), torch.stack(x_mv, 0), torch.stack(gt_bev, 0), torch.stack(gt_aux, 0)

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate,
        drop_last=True,
    )
    steps_per_epoch = max(1, len(loader))
    print("[DATA] len(ds)=", len(ds), "len(loader)=", len(loader), "steps/epoch=", steps_per_epoch)

    model = MVDetMultiCam(
        num_views=len(views),
        bev_hw=(Hg, Wg),
        grids=grids,
        valids=valids,
        reduce_ch=args.reduce_ch,
        pretrained=args.pretrained,
    ).to(dev)

    if args.freeze_bn:
        freeze_batchnorm(model)
        print("[CFG] freeze_bn = True")

    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and dev.type == "cuda"))
    save_steps = set(int(s.strip()) for s in args.save_steps.split(",") if s.strip().isdigit())
    mse = nn.MSELoss(reduction="mean")

    # fixed sample for saving
    fixed_item = None
    if args.fixed_stem:
        ann = data_root / "annotations_positions" / f"{args.fixed_stem}.json"
        if ann.exists():
            for i in range(len(ds)):
                stem, x_mv, gt_bev, gt_aux = ds[i]
                if stem == args.fixed_stem:
                    fixed_item = (stem, x_mv, gt_bev, gt_aux)
                    print("[FIXED] using stem =", stem)
                    break
        if fixed_item is None:
            print("[FIXED] not found:", args.fixed_stem)

    global_step = 0
    model.train()

    for ep in range(args.epochs):
        for stems, x_mv, gt_bev, gt_aux in loader:
            x_mv = x_mv.to(dev, non_blocking=True)     # (B,V,3,Hi,Wi)
            gt_bev = gt_bev.to(dev, non_blocking=True) # (B,1,Hg,Wg)
            gt_aux = gt_aux.to(dev, non_blocking=True) # (B,V,2,Hs,Ws)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                pred_bev, pred_aux = model(x_mv)

                # ground-plane loss (MSE), optional pos-region weighting
                diff2 = (pred_bev - gt_bev) ** 2
                if args.loss_pos_w != 1.0:
                    pos_mask = (gt_bev > args.pos_thr).float()
                    w = 1.0 + (args.loss_pos_w - 1.0) * pos_mask
                    loss_ground = (diff2 * w).mean()
                else:
                    loss_ground = diff2.mean()

                # aux loss across views/channels
                loss_aux = mse(pred_aux, gt_aux)

                loss = args.ground_w * loss_ground + args.aux_w * loss_aux

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    lr = sched.get_last_lr()[0]
                    pos_mask = (gt_bev > args.pos_thr)
                    pos_mse = diff2[pos_mask].mean().item() if pos_mask.any() else float("nan")

                    pr_min = pred_bev.min().item()
                    pr_max = pred_bev.max().item()
                    pr_mean = pred_bev.mean().item()
                    mx_gt = gt_bev.max().item()

                    # aux quick stats
                    aux_diff2 = (pred_aux - gt_aux) ** 2
                    aux_pos = (gt_aux > args.pos_thr)
                    aux_pos_mse = aux_diff2[aux_pos].mean().item() if aux_pos.any() else float("nan")

                    print(
                        f"[ep {ep} step {global_step}] "
                        f"loss={loss.item():.6f} ground={loss_ground.item():.6f} aux={loss_aux.item():.6f} "
                        f"pos_mse={pos_mse:.6f} aux_pos_mse={aux_pos_mse:.6f} "
                        f"pred_bev_raw=[{pr_min:.3f},{pr_max:.3f}] mean={pr_mean:.3f} max_gt={mx_gt:.3f} lr={lr:.5f}"
                    )

            if global_step in save_steps:
                with torch.no_grad():
                    if fixed_item is not None:
                        fstem, fx_mv, fgt_bev, _fgt_aux = fixed_item
                        fx_mv = fx_mv.unsqueeze(0).to(dev)
                        fpred_bev, _fpred_aux = model(fx_mv)
                        p = fpred_bev[0, 0].detach().cpu().numpy()
                        g = fgt_bev[0].detach().cpu().numpy()
                        if args.viz_sigmoid:
                            p = 1.0 / (1.0 + np.exp(-p))
                        save_heat_png(out_dir / f"{fstem}_bev_pred_step{global_step}.png", p)
                        save_heat_png(out_dir / f"{fstem}_bev_gt_step{global_step}.png", g)
                    else:
                        p = pred_bev[0, 0].detach().cpu().numpy()
                        g = gt_bev[0, 0].detach().cpu().numpy()
                        if args.viz_sigmoid:
                            p = 1.0 / (1.0 + np.exp(-p))
                        save_heat_png(out_dir / f"{stems[0]}_bev_pred_step{global_step}.png", p)
                        save_heat_png(out_dir / f"{stems[0]}_bev_gt_step{global_step}.png", g)

            global_step += 1

    ckpt = out_dir / "model_multicam_mvdet_style.pth"
    torch.save(model.state_dict(), ckpt)
    print("[OK] saved:", ckpt)


if __name__ == "__main__":
    main()
