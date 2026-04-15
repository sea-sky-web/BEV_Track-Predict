# scripts/utils.py
"""
工具函数模块：图像保存、核函数构建等
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def save_heat_png(path: Path, arr: np.ndarray) -> None:
    """
    将热图数组保存为 PNG 图像
    
    对输入数组进行以下处理：
    1. 转换为 float32
    2. 归一化到 [0, 1]
    3. 缩放到 [0, 255] 并保存
    
    Args:
        path: 输出文件路径
        arr: 热图数组，任意数值范围
        
    Returns:
        None
        
    Example:
        >>> import numpy as np
        >>> heatmap = np.random.rand(256, 256)
        >>> save_heat_png(Path("output.png"), heatmap)
    """
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(str(path))


def build_gaussian_kernel_2d(
    ksize: int,
    sigma: float,
    device: torch.device
) -> torch.Tensor:
    """
    构建二维高斯卷积核
    
    用于热图平滑处理，对标 GaussianMSE 损失函数的高斯模糊步骤。
    
    Args:
        ksize: 核大小（必须为奇数），例如 11, 13, 15
        sigma: 高斯标准差，控制模糊程度
               - 典型值 2.0-3.0
               - 值越大，模糊越强
        device: PyTorch 设备（cuda 或 cpu）
        
    Returns:
        torch.Tensor: 高斯核，形状为 (1, 1, ksize, ksize)
                     
    Raises:
        AssertionError: 如果 ksize 不是奇数
        
    Example:
        >>> dev = torch.device("cuda")
        >>> kernel = build_gaussian_kernel_2d(ksize=11, sigma=2.5, device=dev)
        >>> print(kernel.shape)
        torch.Size([1, 1, 11, 11])
        >>> print(kernel.sum().item())  # 应接近 1.0
        1.0
    """
    assert ksize % 2 == 1, f"ksize 必须为奇数，得到 {ksize}"
    
    r = ksize // 2
    
    # 在设备上创建坐标网格
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    ys = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    
    # 创建网格
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    
    # 计算高斯函数：exp(-(x² + y²) / (2σ²))
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    
    # 归一化使其和为 1
    g = g / (g.max().clamp_min(1e-12))
    
    # 返回形状为 (1, 1, ksize, ksize) 的核
    return g.view(1, 1, ksize, ksize)


def apply_gaussian_blur(
    heatmap: torch.Tensor,
    kernel: torch.Tensor,
    pad_mode: str = "replicate"
) -> torch.Tensor:
    """
    对热图应用高斯模糊
    
    Args:
        heatmap: 输入热图 (B, C, H, W)
        kernel: 高斯核 (1, 1, ksize, ksize)
        pad_mode: 填充模式 ("replicate", "zeros", "reflect" 等)
        
    Returns:
        torch.Tensor: 模糊后的热图，形状同输入
        
    Example:
        >>> heatmap = torch.randn(2, 1, 64, 64)
        >>> kernel = build_gaussian_kernel_2d(11, 2.5, torch.device("cpu"))
        >>> blurred = apply_gaussian_blur(heatmap, kernel)
        >>> print(blurred.shape)
        torch.Size([2, 1, 64, 64])
    """
    B, C, H, W = heatmap.shape
    kernel = kernel.to(dtype=heatmap.dtype, device=heatmap.device)
    pad = (kernel.shape[-1] - 1) // 2
    
    # 对每个通道独立进行卷积
    heatmap_flat = heatmap.reshape(B * C, 1, H, W)
    blurred = F.conv2d(heatmap_flat, kernel, padding=pad, padding_mode=pad_mode)
    blurred = blurred.reshape(B, C, H, W)
    
    return blurred
