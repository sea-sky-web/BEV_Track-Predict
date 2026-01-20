# scripts/models.py
"""
神经网络模型模块：backbone、heads 和完整网络架构
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from geometry import warp_perspective_torch


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
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        
        # 创建 ResNet50 并应用 dilated convolutions
        # replace_stride_with_dilation=[False, True, True]
        # layer2 和 layer3 使用空洞卷积
        m = torchvision.models.resnet50(
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
    
    def __init__(self, in_ch: int = 512, mid_ch: int = 128):
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


class MVDetLikeNet(nn.Module):
    """
    MVDet 风格的多视角 BEV 检测网络
    
    架构：
    1. 多视角特征提取：ResNet50 主干 × V 视角
    2. 单视角预测：head/foot 热图预测（辅助任务）
    3. 投影与融合：透视变换投影到 BEV + 拼接
    4. BEV 预测：融合后的特征预测 BEV 热图
    
    输出：
    - map_logits: BEV 热图 (B, 1, Hb, Wb)
    - imgs_logits: 图像热图 (B, V, 2, Hf, Wf)
    
    Attributes:
        backbone: 共享的 ResNet50 主干
        img_head: 图像预测头
        proj_mats: 投影矩阵参数
        coord: 可学习的坐标编码（可选）
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
        add_coord: bool = True,
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
            add_coord: 是否添加坐标编码，默认 True
        """
        super().__init__()
        
        self.V = num_views
        self.Hb, self.Wb = reduced_hw
        self.Hf, self.Wf = feat_hw
        self.add_coord = add_coord
        
        # 共享的特征提取主干
        self.backbone = ResNet50Stride8Trunk(pretrained=pretrained, out_ch=feat_ch)
        
        # 单视角预测头
        self.img_head = ImgHeadFoot(in_ch=feat_ch, mid_ch=128)
        
        # 投影矩阵（不参与优化）
        self.proj_mats = nn.Parameter(proj_mats, requires_grad=False)
        
        # 计算 BEV 融合特征的输入通道数
        in_bev = num_views * feat_ch
        
        # 可选的坐标编码
        if add_coord:
            in_bev += 2
            
            # 创建坐标网格 [-1, 1]
            xs = torch.linspace(-1, 1, self.Hb).view(self.Hb, 1).expand(self.Hb, self.Wb)
            ys = torch.linspace(-1, 1, self.Wb).view(1, self.Wb).expand(self.Hb, self.Wb)
            coord = torch.stack([ys, xs], dim=0).unsqueeze(0)  # (1, 2, Hb, Wb)
            
            self.coord = nn.Parameter(coord, requires_grad=False)
        else:
            self.coord = None
        
        # BEV 融合预测头
        self.bev_head = BEVHeadDilated(in_ch=in_bev, mid_ch=256)
    
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
        
        # 拼接多视角 BEV 特征
        bev_cat = torch.cat(feats_bev, dim=1)  # (B, V*feat_ch, Hb, Wb)
        
        # 添加坐标编码
        if self.add_coord:
            coord = self.coord.to(bev_cat.device).expand(B, -1, -1, -1)
            bev_cat = torch.cat([bev_cat, coord], dim=1)
        
        # BEV 融合预测
        map_logits = self.bev_head(bev_cat)  # (B, 1, Hb, Wb)
        
        return map_logits, imgs_logits


def create_model(
    num_views: int,
    proj_mats: torch.Tensor,
    reduced_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
    device: torch.device,
    pretrained: bool = True,
    feat_ch: int = 512,
    add_coord: bool = True,
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
        feat_ch: 特征通道数
        add_coord: 是否添加坐标编码
        
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
        add_coord=add_coord,
    ).to(device)
    
    return model
