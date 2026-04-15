# scripts/07_train_single_cam_occ_mvdet_style.py
# MVDet-style single-camera BEV POM regression on Wildtrack
#
# Key points strictly following MVDet (ECCV'20):
# - Treat occupancy map as regression; output has NO activation.
# - Build soft target f(g) by placing Gaussian kernels on GT footpoints.
# - Loss: L2 / Euclidean distance between pred map and soft target (implemented as MSE).
# - Spatial aggregation head: 3x3 dilated conv with dilation 1/2/4, then 1x1 output.
# - Backbone: ResNet-18 with dilation replacing strides to get stride=8 and C=512 features.
#   (torchvision resnet18 BasicBlock doesn't support dilation>1 => we implement a compatible BasicBlock.)
# - Optim: SGD(momentum=0.5, weight_decay=5e-4) + OneCycleLR.
#
# References: MVDet paper Sec 3.2-3.3, 4.2 implementation details.

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional

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


# ------------------ GT heatmap (soft target) ------------------
def build_gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    xs = np.arange(size, dtype=np.float32) - radius
    ys = np.arange(size, dtype=np.float32) - radius
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    g /= (g.max() + 1e-12)
    return g.astype(np.float32)


def add_gaussian_max(heat: np.ndarray, cx: int, cy: int, kernel: np.ndarray):
    """Place Gaussian kernel at (cx,cy) onto heatmap using max merge (landmark-style)."""
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


class WildtrackSingleCamPOM(Dataset):
    """
    Single-camera input image, BEV POM soft target from annotations_positions (positionID footpoints).
    """
    def __init__(
        self,
        data_root: Path,
        view: int,
        max_frames: int,
        bev_down: int,
        sigma: float,
        img_size: Tuple[int, int],  # (H,W)
        train_split: float = 1.0,   # keep 1.0 by default (you can set 0.9 like MVDet later)
        split: str = "train",       # "train" or "test"
        seed: int = 0,
    ):
        self.data_root = data_root
        self.view = view
        self.bev_down = bev_down
        self.img_H, self.img_W = img_size

        self.ann_dir = data_root / "annotations_positions"
        self.img_dir = data_root / "Image_subsets" / f"C{view+1}"
        assert self.ann_dir.exists(), f"missing {self.ann_dir}"
        assert self.img_dir.exists(), f"missing {self.img_dir}"

        ann_files = sorted(self.ann_dir.glob("*.json"))
        if max_frames > 0:
            ann_files = ann_files[:max_frames]
        assert len(ann_files) > 0, "no annotation files found"

        # Optional deterministic split
        if train_split < 1.0:
            rng = np.random.RandomState(seed)
            idx = np.arange(len(ann_files))
            rng.shuffle(idx)
            cut = int(len(idx) * train_split)
            if split == "train":
                idx = idx[:cut]
            else:
                idx = idx[cut:]
            ann_files = [ann_files[i] for i in idx]

        self.ann_files = ann_files

        # MVDet: down=4 -> 120x360 on Wildtrack
        self.Hg = NB_WIDTH // bev_down
        self.Wg = NB_HEIGHT // bev_down

        radius = int(np.ceil(3 * sigma))
        self.kernel = build_gaussian_kernel(sigma=sigma, radius=radius)

        # ImageNet normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.ann_files)

    def _load_image(self, stem: str) -> Image.Image:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"missing image for {stem} in {self.img_dir}")

    def _load_soft_gt(self, ann_path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        heat = np.zeros((self.Hg, self.Wg), dtype=np.float32)
        pts: List[Tuple[int, int]] = []

        for obj in data:
            pos_id = obj.get("positionID", None)
            if pos_id is None:
                continue
            pos_id = int(pos_id)

            ix = pos_id % NB_WIDTH
            iy = pos_id // NB_WIDTH

            gx = ix // self.bev_down    # 0..Hg-1  (rows)
            gy = iy // self.bev_down    # 0..Wg-1  (cols)

            if 0 <= gx < self.Hg and 0 <= gy < self.Wg:
                pts.append((gx, gy))
                add_gaussian_max(heat, gx, gy, self.kernel)

        return heat, pts

    def __getitem__(self, idx: int):
        ann_path = self.ann_files[idx]
        stem = ann_path.stem

        img = self._load_image(stem).resize((self.img_W, self.img_H), Image.BILINEAR)
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).float() / 255.0  # (H,W,3)
        x = x.permute(2, 0, 1)  # (3,H,W)
        x = (x - self.mean) / self.std

        heat, pts = self._load_soft_gt(ann_path)
        gt = torch.from_numpy(heat).unsqueeze(0)  # (1,Hg,Wg) in [0,1]

        return stem, x, gt, pts


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
    """
    BEV cell center (world z=0) -> project -> image -> map to feature coords -> normalize for grid_sample.
    grid_sample align_corners=True (consistent with normalization below).
    """
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

    # image -> feature coordinate (assuming feature is bilinear-resized from backbone output)
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


# ------------------ ResNet18 with dilation support ------------------
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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Stride8Dilated(nn.Module):
    """
    ResNet-18 backbone that outputs stride=8 feature map with C=512 by replacing
    layer3/layer4 strides with dilation (2 and 4), matching MVDet description.
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
        self.layer2 = self._make_layer(128, blocks=2, stride=2, dilate=False)  # stride=8 here

        # For ResNet: replace_stride_with_dilation corresponds to layer2/3/4 in torchvision;
        # Here we apply it to layer3 and layer4 (MVDet wants keep stride=8).
        self.layer3 = self._make_layer(256, blocks=2, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, blocks=2, stride=2, dilate=replace_stride_with_dilation[2])

        # No avgpool/fc.

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
            # replace stride with dilation
            self.dilation *= stride
            stride = 1
        dilation = self.dilation

        if stride != 1 or self.inplanes != planes * BasicBlockDilated.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlockDilated.expansion, stride),
                norm_layer(planes * BasicBlockDilated.expansion),
            )

        layers = []
        layers.append(
            BasicBlockDilated(
                self.inplanes, planes, stride=stride, downsample=downsample,
                dilation=prev_dilation if not dilate else dilation, norm_layer=norm_layer
            )
        )
        self.inplanes = planes * BasicBlockDilated.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlockDilated(
                    self.inplanes, planes, stride=1, downsample=None,
                    dilation=dilation, norm_layer=norm_layer
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # (B,3,720,1280)
        x = self.conv1(x)   # stride2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # stride4

        x = self.layer1(x)  # stride4
        x = self.layer2(x)  # stride8
        x = self.layer3(x)  # stride8 (dilated)
        x = self.layer4(x)  # stride8 (more dilated)
        return x  # (B,512,H/8,W/8)


def build_resnet18_stride8_dilated(pretrained: bool) -> ResNet18Stride8Dilated:
    m = ResNet18Stride8Dilated(replace_stride_with_dilation=(False, True, True))
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        tv = torchvision.models.resnet18(weights=weights)
        sd = tv.state_dict()
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # Missing/unexpected are acceptable (we removed fc/avgpool).
        _ = (missing, unexpected)
    return m


# ------------------ BEV Head & Full Model ------------------
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

            nn.Conv2d(mid_ch, 1, 1, bias=True),  # NO activation (MVDet)
        )

    def forward(self, x):
        return self.net(x)  # (B,1,Hg,Wg) raw regression output


class MVDetSingleCamPOM(nn.Module):
    def __init__(self, pretrained: bool, bev_hw: Tuple[int, int], grid: torch.Tensor, valid: torch.Tensor):
        super().__init__()
        self.backbone = build_resnet18_stride8_dilated(pretrained=pretrained)

        self.grid = nn.Parameter(grid, requires_grad=False)
        self.valid = nn.Parameter(valid, requires_grad=False)

        Hg, Wg = bev_hw
        xs = torch.linspace(-1, 1, Hg).view(Hg, 1).expand(Hg, Wg)
        ys = torch.linspace(-1, 1, Wg).view(1, Wg).expand(Hg, Wg)
        coord = torch.stack([xs, ys], dim=0).unsqueeze(0)  # (1,2,Hg,Wg)
        self.coord = nn.Parameter(coord, requires_grad=False)

        # MVDet uses C=512 + coord(2)
        self.head = BEVHeadDilated(in_ch=512 + 2, mid_ch=256)

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)  # (B,512,Hi/8,Wi/8)

        # MVDet: interpolate to fixed Hf,Wf before projection
        f = F.interpolate(f, size=(270, 480), mode="bilinear", align_corners=False)

        B = f.shape[0]
        grid = self.grid.expand(B, -1, -1, -1)
        valid = self.valid.expand(B, -1, -1, -1)

        bev = F.grid_sample(f, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        bev = bev * valid

        coord = self.coord.to(bev.device).expand(B, -1, -1, -1)
        bev = torch.cat([bev, coord], dim=1)

        out = self.head(bev)  # raw regression
        return out


# ------------------ Training utils ------------------
def freeze_batchnorm(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False


def save_heat_png(path: Path, arr: np.ndarray):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="wildtrack")
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--bev_down", type=int, default=4)

    parser.add_argument("--sigma", type=float, default=3.0, help="GT Gaussian sigma on 120x360 grid (cells)")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_lr", type=float, default=0.1, help="MVDet uses 0.1 (adjust lower for weak GPU)")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--freeze_bn", action="store_true", help="freeze BN stats & params (often stabilizes batch=1)")
    parser.add_argument("--loss_pos_w", type=float, default=1.0, help="optional positive-region weight (default=1, MVDet)")
    parser.add_argument("--pos_thr", type=float, default=0.1, help="positive region threshold for weighting/logging")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_steps", type=str, default="0,50,100")
    parser.add_argument("--fixed_stem", type=str, default="", help="e.g. 00000055")
    parser.add_argument("--viz_sigmoid", action="store_true", help="only for visualization: apply sigmoid before saving")

    # optional split like MVDet (0.9 train / 0.1 test). Keep default 1.0 for your current workflow.
    parser.add_argument("--train_split", type=float, default=1.0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path("outputs") / "train_single_cam_occ_mvdet_style" / "model_single_cam_occ_mvdet_style"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("You requested cuda but torch.cuda.is_available() is False.")
    dev = torch.device(args.device)
    print("[DEV] torch.cuda.is_available =", torch.cuda.is_available())
    print("[DEV] selected device =", dev)
    if dev.type == "cuda":
        print("[DEV] gpu name =", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # MVDet resize and fixed feature size
    Hi, Wi = 720, 1280
    Hf, Wf = 270, 480

    Hg = NB_WIDTH // args.bev_down
    Wg = NB_HEIGHT // args.bev_down
    print(f"[CFG] img={Hi}x{Wi}, feat={Hf}x{Wf}, bev={Hg}x{Wg}, bev_down={args.bev_down}, sigma={args.sigma}")

    # calib
    calib_root = data_root / "calibrations"
    cam_name = CAM_NAMES[args.view]
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

    print("[CALIB] cam =", cam_name)
    print("[CALIB] intr =", intr_path)
    print("[CALIB] extr =", extr_path)

    grid, valid = build_bev_sampling_grid(
        K=K,
        rvec=rvec,
        tvec=tvec,
        img_hw=(Hi, Wi),
        feat_hw=(Hf, Wf),
        bev_hw=(Hg, Wg),
        bev_down=args.bev_down,
        device=dev,
    )
    print(f"[GRID] valid_ratio={(valid.mean().item()):.4f}")

    ds = WildtrackSingleCamPOM(
        data_root=data_root,
        view=args.view,
        max_frames=args.max_frames,
        bev_down=args.bev_down,
        sigma=args.sigma,
        img_size=(Hi, Wi),
        train_split=args.train_split,
        split=args.split,
        seed=args.seed,
    )

    def collate(batch):
        stems, xs, gts, pts = zip(*batch)
        x = torch.stack(xs, dim=0)
        gt = torch.stack(gts, dim=0)
        return list(stems), x, gt, list(pts)

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True if args.split == "train" else False,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate,
        drop_last=True if args.split == "train" else False,
    )

    model = MVDetSingleCamPOM(pretrained=args.pretrained, bev_hw=(Hg, Wg), grid=grid, valid=valid).to(dev)
    if args.freeze_bn:
        freeze_batchnorm(model)
        print("[CFG] freeze_bn = True")
    print("[DEV] model param device =", next(model.parameters()).device)

    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=5e-4)
    steps_per_epoch = max(1, len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )

    save_steps = set(int(s.strip()) for s in args.save_steps.split(",") if s.strip().isdigit())

    # fixed frame for consistent visualization
    fixed_item = None
    if args.fixed_stem:
        ann = data_root / "annotations_positions" / f"{args.fixed_stem}.json"
        if ann.exists():
            for i in range(len(ds)):
                stem, _, _, _ = ds[i]
                if stem == args.fixed_stem:
                    fixed_item = ds[i]
                    print("[FIXED] using stem =", args.fixed_stem)
                    break
        if fixed_item is None:
            print("[FIXED] not found stem:", args.fixed_stem)

    global_step = 0
    model.train()

    for ep in range(args.epochs):
        for stems, x, gt, _pts_list in loader:
            x = x.to(dev, non_blocking=True)
            gt = gt.to(dev, non_blocking=True)  # soft target in [0,1]

            pred = model(x)  # raw output (no activation)

            # MVDet loss: L2 between pred and f(g) (soft target). Implement as MSE.
            diff2 = (pred - gt) ** 2

            if args.loss_pos_w != 1.0:
                pos_mask = (gt > args.pos_thr).float()
                w = 1.0 + (args.loss_pos_w - 1.0) * pos_mask
                loss = (diff2 * w).mean()
            else:
                loss = diff2.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    lr = sched.get_last_lr()[0]
                    pos_mask = (gt > args.pos_thr)
                    pos_mse = diff2[pos_mask].mean().item() if pos_mask.any() else float("nan")
                    pred_min = pred.min().item()
                    pred_max = pred.max().item()
                    pred_mean = pred.mean().item()
                    max_gt = gt.max().item()
                    print(
                        f"[ep {ep} step {global_step}] "
                        f"loss={loss.item():.6f} pos_mse={pos_mse:.6f} "
                        f"pred_raw=[{pred_min:.3f},{pred_max:.3f}] mean={pred_mean:.3f} "
                        f"max_gt={max_gt:.3f} lr={lr:.5f}"
                    )

            if global_step in save_steps:
                with torch.no_grad():
                    if fixed_item is not None:
                        fstem, fx, fgt, _fpts = fixed_item
                        fx = fx.unsqueeze(0).to(dev)
                        fpred = model(fx)[0, 0].detach().cpu().numpy()
                        fgt_np = fgt[0].detach().cpu().numpy()

                        if args.viz_sigmoid:
                            fpred_v = 1.0 / (1.0 + np.exp(-fpred))
                        else:
                            fpred_v = fpred

                        save_heat_png(out_dir / f"{fstem}_pred_step{global_step}.png", fpred_v)
                        save_heat_png(out_dir / f"{fstem}_gt_step{global_step}.png", fgt_np)
                    else:
                        p0 = pred[0, 0].detach().cpu().numpy()
                        g0 = gt[0, 0].detach().cpu().numpy()
                        if args.viz_sigmoid:
                            p0 = 1.0 / (1.0 + np.exp(-p0))
                        save_heat_png(out_dir / f"{stems[0]}_pred_step{global_step}.png", p0)
                        save_heat_png(out_dir / f"{stems[0]}_gt_step{global_step}.png", g0)

            global_step += 1

    torch.save(model.state_dict(), out_dir / "model_single_cam_mvdet_style.pth")
    print("[OK] saved model:", out_dir / "model_single_cam_mvdet_style.pth")


if __name__ == "__main__":
    main()
