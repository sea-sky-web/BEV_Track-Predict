# src/models.py
"""
神经网络模型模块：backbone、heads 和完整网络架构
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from geometry import warp_perspective_torch

BackboneName = Literal["resnet18", "resnet50", "mobilenet_v2"]
FusionMode = Literal["concat", "confidence", "confidence_v1", "confidence_v2", "geo_confidence_v1"]


def normalize_fusion_mode(fusion_mode: str) -> str:
    """Keep legacy CLI aliases while making the active attention fusion explicit."""
    if fusion_mode == "confidence":
        return "confidence_v2"
    return fusion_mode


def _apply_mvdet_dilation(layer: nn.Sequential, previous_dilation: int, new_dilation: int) -> None:
    """Apply MVDet's progressive dilation pattern to a ResNet-18 layer.

    MVDet's custom BasicBlock applies dilation ONLY to conv1.
    Within a layer, the first block keeps the previous layer's dilation,
    subsequent blocks use the new (larger) dilation.

    For layer3 (dilate=True, original stride=2):
      block0: conv1 dilation=previous(1), stride=1
      block1: conv1 dilation=new(2)

    For layer4 (dilate=True, original stride=2):
      block0: conv1 dilation=previous(2), stride=1
      block1: conv1 dilation=new(4)
    """
    blocks = list(layer.children())
    for i, block in enumerate(blocks):
        if hasattr(block, "conv1"):
            if block.conv1.stride != (1, 1):
                block.conv1.stride = (1, 1)
            d = previous_dilation if i == 0 else new_dilation
            block.conv1.dilation = (d, d)
            block.conv1.padding = (d, d)
        if getattr(block, "downsample", None) is not None:
            conv = block.downsample[0]
            if hasattr(conv, "stride") and conv.stride != (1, 1):
                conv.stride = (1, 1)


def _find_depthwise_conv(block: nn.Module) -> nn.Conv2d | None:
    """Find the depthwise Conv2d inside a MobileNetV2 InvertedResidual block."""
    for module in block.modules():
        if isinstance(module, nn.Conv2d) and module.groups == module.in_channels and module.kernel_size != (1, 1):
            return module
    return None


def _apply_mobilenet_dilation(blocks: list[nn.Module], dilation: int) -> None:
    """Replace stride-2 with dilation in MobileNetV2 InvertedResidual blocks.

    The first block with stride=2 gets stride→1 and the new dilation.
    All subsequent blocks also get the new dilation on their depthwise conv.
    """
    for block in blocks:
        dw = _find_depthwise_conv(block)
        if dw is None:
            continue
        if dw.stride != (1, 1):
            dw.stride = (1, 1)
        dw.dilation = (dilation, dilation)
        dw.padding = (dilation, dilation)


class MobileNetV2Stride8Trunk(nn.Module):
    """MobileNet-V2 backbone with stride-8 output via dilated convolutions.

    Truncated at features[13] (96ch) to keep memory feasible for multi-view
    training with gradient computation. Only features[7:14] use dilation=2;
    the expensive stride-32 layers (features[14:18], expansion=6 ×160/320ch)
    are dropped to avoid OOM on 15GB GPUs with 7 views.

    features[0:7]  → natural stride 8  (32ch)
    features[7:14] → dilation=2, stride 8  (96ch)
    reduce         → 1x1 conv 96→out_ch
    """

    def __init__(self, pretrained: bool = True, out_ch: int = 512):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v2(weights=weights)

        features = list(m.features.children())

        self.features_s8 = nn.Sequential(*features[:7])

        s16_blocks = features[7:14]
        _apply_mobilenet_dilation(s16_blocks, dilation=2)
        self.features_d2 = nn.Sequential(*s16_blocks)

        self.reduce = nn.Conv2d(96, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_s8(x)
        x = self.features_d2(x)
        return self.reduce(x)


class ResNet18Stride8Trunk(nn.Module):
    """
    ResNet-18 backbone with stride-8 output, matching the MVDet-style lightweight encoder.
    """

    def __init__(self, pretrained: bool = True, out_ch: int = 512):
        super().__init__()
        if out_ch != 512:
            raise ValueError("ResNet18Stride8Trunk currently outputs 512 channels; set feat_ch=512.")

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)

        # MVDet progressive dilation: layer3 gets (prev=1, new=2), layer4 gets (prev=2, new=4)
        _apply_mvdet_dilation(m.layer3, previous_dilation=1, new_dilation=2)
        _apply_mvdet_dilation(m.layer4, previous_dilation=2, new_dilation=4)

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.reduce = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.reduce(x)


class ResNet50Stride8Trunk(nn.Module):
    """
    ResNet50 主干网络，输出 stride=8 特征
    
    使用 dilated convolution（空洞卷积）代替 stride=2，
    保持空间分辨率，增加感受野。
    
    Attributes:
        stem: conv1 + bn1 + relu + maxpool，stride=4
        layer1-4: ResNet blocks
        reduce: 1x1 卷积缩减通道
        
    References:
        - https://arxiv.org/abs/1512.03385 (ResNet)
        - https://arxiv.org/abs/1706.05587 (dilated convolutions)
    """
    
    def __init__(self, pretrained: bool = True, out_ch: int = 512):
        """
        初始化 ResNet50 主干网络
        
        Args:
            pretrained: 是否加载 ImageNet 预训练权重
            out_ch: 输出通道数，默认 512
        """
        super().__init__()
        
        # 加载预训练权重
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        
        # 创建 ResNet50 并应用 dilated convolutions
        # replace_stride_with_dilation=[False, True, True]
        # layer2 和 layer3 使用空洞卷积
        m = resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, True, True]
        )
        
        # 分解为各部分
        self.stem = nn.Sequential(
            m.conv1,      # 7x7 conv，stride=2
            m.bn1,        # BatchNorm
            m.relu,       # ReLU
            m.maxpool     # 3x3 maxpool，stride=2
        )
        self.layer1 = m.layer1  # stride=4 (2x2 from stem+maxpool)
        self.layer2 = m.layer2  # stride=8 (2x2 dilation)
        self.layer3 = m.layer3  # stride=8 (dilation=2)
        self.layer4 = m.layer4  # stride=8 (dilation=4)
        
        # 输出通道约简
        self.reduce = nn.Conv2d(2048, out_ch, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            torch.Tensor: 特征图 (B, out_ch, H/8, W/8)
        """
        x = self.stem(x)        # H/4, W/4
        x = self.layer1(x)      # H/4, W/4, 256 channels
        x = self.layer2(x)      # H/8, W/8, 512 channels
        x = self.layer3(x)      # H/8, W/8, 1024 channels (dilation=2)
        x = self.layer4(x)      # H/8, W/8, 2048 channels (dilation=4)
        x = self.reduce(x)      # H/8, W/8, out_ch channels
        return x


class ImgHeadFoot(nn.Module):
    """
    图像特征平面预测头
    
    从特征图预测两个通道的热图：
    - 通道 0: 人头位置
    - 通道 1: 人脚位置
    
    这些单视角预测用作辅助监督，约束单视角特征学习。
    """
    
    def __init__(self, in_ch: int = 512, mid_ch: int = 64):
        """
        初始化图像预测头
        
        Args:
            in_ch: 输入通道数
            mid_ch: 中间层通道数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 2, 1)  # 2 通道输出 (head, foot)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 特征图 (B, in_ch, H, W)
            
        Returns:
            torch.Tensor: logits (B, 2, H, W)
                         - [:, 0]: 人头热图
                         - [:, 1]: 人脚热图
        """
        return self.net(x)


class MVDetMapClassifier(nn.Module):
    """MVDet 原始论文的 BEV head：3 层 dilated conv，无 BatchNorm，输出层无 bias。

    Reference: github.com/hou-yz/MVDet/blob/master/multiview_detector/models/persp_trans_detector.py
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVHeadDilated(nn.Module):
    """
    BEV 融合预测头，使用 dilated convolutions
    
    从多视角融合特征预测 BEV 热图。
    使用递增的 dilation 扩大感受野，捕捉长距离上下文。
    
    Dilation 系列: [1, 2, 4]
    """
    
    def __init__(self, in_ch: int, mid_ch: int = 256):
        """
        初始化 BEV 预测头
        
        Args:
            in_ch: 输入通道数（多视角特征拼接 + 坐标）
            mid_ch: 中间层通道数
        """
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
            
            nn.Conv2d(mid_ch, 1, 1)  # 1 通道输出
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 融合特征 (B, in_ch, H, W)
            
        Returns:
            torch.Tensor: logits (B, 1, H, W)
        """
        return self.net(x)


class SpatialAwareConfidenceFusion(nn.Module):
    """Learn BEV-space per-view confidence weights and fuse projected features."""

    def __init__(self, feat_ch: int, hidden_ch: int = 64):
        super().__init__()
        hidden_ch = max(16, min(hidden_ch, feat_ch // 4))
        self.score_net = nn.Sequential(
            nn.Conv2d(feat_ch, hidden_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )

    def forward(self, feats_bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats_bev: projected per-view BEV features (B, V, C, Hb, Wb)

        Returns:
            Fused BEV feature map (B, C, Hb, Wb)
        """
        if feats_bev.ndim != 5:
            raise ValueError(f"Expected (B,V,C,H,W) BEV features, got {tuple(feats_bev.shape)}")

        b, v, c, h, w = feats_bev.shape
        flat = feats_bev.reshape(b * v, c, h, w)
        scores = self.score_net(flat).reshape(b, v, 1, h, w)
        weights = torch.softmax(scores, dim=1)
        return (weights * feats_bev).sum(dim=1)


class ConcatAttentionFusion(nn.Module):
    """Predict per-view BEV weights from the joint multi-view representation."""

    def __init__(self, num_views: int, feat_ch: int):
        super().__init__()
        self.num_views = num_views
        self.joint_compress = nn.Sequential(
            nn.Conv2d(num_views * feat_ch, feat_ch, 1),
            nn.ReLU(inplace=True),
        )
        self.weight_head = nn.Conv2d(feat_ch, num_views, 1)
        self.latest_weights = None

    def forward(self, feats_bev: torch.Tensor) -> torch.Tensor:
        if feats_bev.ndim != 5:
            raise ValueError(f"Expected (B,V,C,H,W) BEV features, got {tuple(feats_bev.shape)}")
        b, v, c, h, w = feats_bev.shape
        if v != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {v}")

        joint = feats_bev.reshape(b, v * c, h, w)
        joint = self.joint_compress(joint)
        weights = torch.softmax(self.weight_head(joint), dim=1)
        self.latest_weights = weights.detach()
        return (feats_bev * weights.unsqueeze(2)).sum(dim=1)


class GeoConfidenceFusion(nn.Module):
    """Geometry-reliability prompted fusion.

    Extends ConcatAttentionFusion with a static geometry prior derived from
    camera projection metadata (valid mask, border margin, coverage count).
    """

    def __init__(self, num_views: int, feat_ch: int, geo_ch: int = 3):
        super().__init__()
        self.num_views = num_views
        self.joint_compress = nn.Sequential(
            nn.Conv2d(num_views * feat_ch, feat_ch, 1),
            nn.ReLU(inplace=True),
        )
        self.feature_weight_head = nn.Conv2d(feat_ch, num_views, 1)
        self.geo_score_net = nn.Conv2d(geo_ch, 1, 1)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.latest_weights = None

    def forward(
        self,
        feats_bev: torch.Tensor,
        geo_meta: torch.Tensor,
    ) -> torch.Tensor:
        b, v, c, h, w = feats_bev.shape

        joint = feats_bev.reshape(b, v * c, h, w)
        joint = self.joint_compress(joint)
        feature_scores = self.feature_weight_head(joint)  # (B, V, H, W)

        geo_scores = torch.stack(
            [self.geo_score_net(geo_meta[vi:vi+1]).squeeze(0) for vi in range(v)],
            dim=0,
        ).squeeze(1).unsqueeze(0)  # (1, V, H, W)

        combined = feature_scores + self.beta * geo_scores
        weights = torch.softmax(combined, dim=1)
        self.latest_weights = weights.detach()

        return (feats_bev * weights.unsqueeze(2)).sum(dim=1)


class MVDetLikeNet(nn.Module):
    """
    MVDet 风格的多视角 BEV 检测网络
    
    架构：
    1. 多视角特征提取：ResNet-18/ResNet-50 共享主干 × V 视角
    2. 单视角预测：head/foot 热图预测（辅助任务）
    3. 投影与融合：透视变换投影到 BEV + 拼接
    4. BEV 预测：融合后的特征预测 BEV 热图
    
    输出：
    - map_logits: BEV 热图 (B, 1, Hb, Wb)
    - imgs_logits: 图像热图 (B, V, 2, Hf, Wf)
    
    Attributes:
        backbone: 共享的 ResNet 主干
        img_head: 图像预测头
        proj_mats: 静态投影矩阵 buffer
        coord: 静态坐标编码 buffer（可选）
        bev_head: BEV 预测头
    """
    
    def __init__(
        self,
        num_views: int,
        proj_mats: torch.Tensor,
        reduced_hw: Tuple[int, int],
        feat_hw: Tuple[int, int],
        feat_ch: int = 512,
        pretrained: bool = True,
        backbone: BackboneName = "resnet18",
        add_coord: bool = True,
        fusion_mode: FusionMode = "confidence_v2",
    ):
        """
        初始化 MVDetLikeNet
        
        Args:
            num_views: 视角数量
            proj_mats: 投影矩阵 (V, 3, 3)，特征平面 -> BEV 网格
            reduced_hw: BEV 网格大小 (Hb, Wb)
            feat_hw: 特征平面大小 (Hf, Wf)
            feat_ch: 特征通道数，默认 512
            pretrained: 是否加载预训练权重，默认 True
            backbone: 主干网络，resnet18 默认，resnet50 保留 legacy 复现实验
            add_coord: 是否添加坐标编码，默认 True
            fusion_mode: concat、confidence_v1 或 confidence_v2
        """
        super().__init__()
        fusion_mode = normalize_fusion_mode(fusion_mode)
        if fusion_mode not in {"concat", "confidence_v1", "confidence_v2", "geo_confidence_v1"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if backbone not in {"resnet18", "resnet50", "mobilenet_v2"}:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.V = num_views
        self.Hb, self.Wb = reduced_hw
        self.Hf, self.Wf = feat_hw
        self.add_coord = add_coord
        self.fusion_mode = fusion_mode
        self.backbone_name = backbone
        
        # 共享的特征提取主干
        if backbone == "resnet18":
            self.backbone = ResNet18Stride8Trunk(pretrained=pretrained, out_ch=feat_ch)
        elif backbone == "mobilenet_v2":
            self.backbone = MobileNetV2Stride8Trunk(pretrained=pretrained, out_ch=feat_ch)
        else:
            self.backbone = ResNet50Stride8Trunk(pretrained=pretrained, out_ch=feat_ch)
        
        # 单视角预测头
        self.img_head = ImgHeadFoot(in_ch=feat_ch, mid_ch=128)
        
        # 投影矩阵（静态 buffer，不参与优化）
        self.register_buffer("proj_mats", proj_mats.detach().clone())

        # 静态几何元数据（仅 geo_confidence 融合模式使用）
        if fusion_mode == "geo_confidence_v1":
            from geometry import compute_bev_geometry_metadata
            geo_meta = compute_bev_geometry_metadata(
                proj_mats=proj_mats,
                src_hw=(self.Hf, self.Wf),
                dst_hw=(self.Hb, self.Wb),
            )
            self.register_buffer("geo_meta", geo_meta)
        else:
            self.geo_meta = None
        
        # 计算 BEV 融合特征的输入通道数
        if fusion_mode == "concat":
            in_bev = num_views * feat_ch
            self.confidence_fusion = None
        elif fusion_mode == "confidence_v1":
            in_bev = feat_ch
            self.confidence_fusion = SpatialAwareConfidenceFusion(feat_ch=feat_ch)
        elif fusion_mode == "geo_confidence_v1":
            in_bev = feat_ch
            self.confidence_fusion = GeoConfidenceFusion(
                num_views=num_views, feat_ch=feat_ch, geo_ch=3,
            )
        else:
            in_bev = feat_ch
            self.confidence_fusion = ConcatAttentionFusion(num_views=num_views, feat_ch=feat_ch)
        
        # 可选的坐标编码
        if add_coord:
            in_bev += 2
            
            # 创建坐标网格 [-1, 1]
            xs = torch.linspace(-1, 1, self.Hb).view(self.Hb, 1).expand(self.Hb, self.Wb)
            ys = torch.linspace(-1, 1, self.Wb).view(1, self.Wb).expand(self.Hb, self.Wb)
            coord = torch.stack([ys, xs], dim=0).unsqueeze(0)  # (1, 2, Hb, Wb)
            
            self.register_buffer("coord", coord)
        else:
            self.coord = None
        
        # BEV 融合预测头
        if fusion_mode == "concat":
            self.bev_head = MVDetMapClassifier(in_ch=in_bev)
        else:
            self.bev_head = BEVHeadDilated(in_ch=in_bev, mid_ch=256)
        
        # Offset 回归头（亚像素精度定位）
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_bev, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )
    
    def forward(self, x_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_views: 多视角输入 (B, V, 3, Hi, Wi)
                    B: 批大小
                    V: 视角数
                    (3, Hi, Wi): RGB 图像
        
        Returns:
            tuple:
                - map_logits: BEV 热图 logits (B, 1, Hb, Wb)
                - imgs_logits: 图像热图 logits (B, V, 2, Hf, Wf)
                              - imgs_logits[:, :, 0]: 人头
                              - imgs_logits[:, :, 1]: 人脚
        """
        B, V, _, _, _ = x_views.shape
        
        feats_bev = []      # 投影后的特征
        imgs_logits = []    # 单视角预测
        
        # 逐视角处理
        for vi in range(V):
            # 特征提取
            f = self.backbone(x_views[:, vi])  # (B, feat_ch, Hi/8, Wi/8)
            
            # 插值到标准特征平面尺寸
            f = F.interpolate(
                f, size=(self.Hf, self.Wf),
                mode="bilinear",
                align_corners=False
            )
            
            # 单视角预测（辅助）
            img_logit = self.img_head(f)  # (B, 2, Hf, Wf)
            imgs_logits.append(img_logit)
            
            # 投影到 BEV
            M = self.proj_mats[vi].unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)
            bev = warp_perspective_torch(f, M, dsize=(self.Hb, self.Wb))  # (B, feat_ch, Hb, Wb)
            feats_bev.append(bev)
        
        # 堆叠单视角预测
        imgs_logits = torch.stack(imgs_logits, dim=1)  # (B, V, 2, Hf, Wf)
        
        if self.fusion_mode == "concat":
            bev_fused = torch.cat(feats_bev, dim=1)  # (B, V*feat_ch, Hb, Wb)
        else:
            bev_stack = torch.stack(feats_bev, dim=1)  # (B, V, feat_ch, Hb, Wb)
            if self.geo_meta is not None:
                bev_fused = self.confidence_fusion(bev_stack, self.geo_meta)
            else:
                bev_fused = self.confidence_fusion(bev_stack)  # (B, feat_ch, Hb, Wb)
        
        # 添加坐标编码
        if self.add_coord:
            coord = self.coord.expand(B, -1, -1, -1)
            bev_fused = torch.cat([bev_fused, coord], dim=1)
        
        # BEV 融合预测
        map_logits = self.bev_head(bev_fused)  # (B, 1, Hb, Wb)
        offset_preds = self.offset_head(bev_fused)  # (B, 2, Hb, Wb)
        
        return map_logits, offset_preds, imgs_logits


def create_model(
    num_views: int,
    proj_mats: torch.Tensor,
    reduced_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
    device: torch.device,
    pretrained: bool = True,
    backbone: BackboneName = "resnet18",
    feat_ch: int = 512,
    add_coord: bool = True,
    fusion_mode: FusionMode = "confidence_v2",
) -> MVDetLikeNet:
    """
    工厂函数：创建完整的 MVDetLikeNet 模型
    
    Args:
        num_views: 视角数量
        proj_mats: 投影矩阵
        reduced_hw: BEV 网格大小
        feat_hw: 特征平面大小
        device: 计算设备
        pretrained: 是否加载预训练权重
        backbone: 主干网络
        feat_ch: 特征通道数
        add_coord: 是否添加坐标编码
        fusion_mode: BEV 融合模式
        
    Returns:
        MVDetLikeNet: 初始化完成的模型
        
    Example:
        >>> model = create_model(
        ...     num_views=3,
        ...     proj_mats=torch.eye(3).unsqueeze(0).repeat(3, 1, 1),
        ...     reduced_hw=(135, 240),
        ...     feat_hw=(270, 480),
        ...     device=torch.device("cuda"),
        ...     pretrained=True
        ... )
    """
    model = MVDetLikeNet(
        num_views=num_views,
        proj_mats=proj_mats,
        reduced_hw=reduced_hw,
        feat_hw=feat_hw,
        feat_ch=feat_ch,
        pretrained=pretrained,
        backbone=backbone,
        add_coord=add_coord,
        fusion_mode=fusion_mode,
    ).to(device)
    
    return model
