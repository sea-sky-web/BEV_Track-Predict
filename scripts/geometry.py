# scripts/geometry.py
"""
几何变换模块：坐标系变换、投影矩阵构建、透视变换等
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from calibration import scale_intrinsics


def make_worldgrid2worldcoord_mat(
    origin_x: float,
    origin_y: float,
    step: float
) -> np.ndarray:
    """
    构建世界网格坐标到世界坐标的变换矩阵
    
    在 MVDet 框架中，BEV 网格是离散的像素网格，
    需要变换到连续的世界坐标系。
    
    变换规则（使用网格中心）：
        X_world = origin_x + (grid_x + 0.5) * step
        Y_world = origin_y + (grid_y + 0.5) * step
    
    矩阵形式（齐次坐标）：
        [X]   [step,  0,    ox] [grid_x]
        [Y] = [ 0,   step, oy] [grid_y]
        [1]   [ 0,    0,    1] [  1   ]
        
    其中 ox = origin_x + 0.5*step, oy = origin_y + 0.5*step
    
    Args:
        origin_x: 世界坐标系 X 原点（米或厘米）
        origin_y: 世界坐标系 Y 原点（米或厘米）
        step: 网格步长（米或厘米，与 origin 单位一致）
        
    Returns:
        np.ndarray: 变换矩阵 (3, 3)，dtype=float64
        
    Example:
        >>> mat = make_worldgrid2worldcoord_mat(-3.0, -9.0, 0.025)
        >>> print(mat.shape)
        (3, 3)
        >>> # 验证网格 (0, 0) 映射到 (-2.9875, -8.9875)
        >>> grid_point = np.array([0, 0, 1])
        >>> world_point = mat @ grid_point
        >>> print(world_point[:2])
        [-2.9875 -8.9875]
    """
    # 使用网格中心：origin + (idx + 0.5) * step
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
    构建 MVDet 风格的投影矩阵
    
    将世界网格坐标映射到图像特征平面坐标。
    
    完整变换链：
        1. 世界网格坐标 (grid_x, grid_y, 1)
        2. → 世界坐标系 (X, Y, 1) via worldgrid2worldcoord
        3. → 相机坐标系 (Xc, Yc, Zc) via [R|t]
        4. → 图像坐标系 (u, v, 1) via K
        5. → 图像特征平面 (u', v', 1) via permutation
    
    数学推导（对标 MVDet 源码）：
        extr = [R | t]  (3x4 矩阵：world to camera)
        extr_3x3 = extr[:, :3]  (移除 Z 列)
        worldcoord2imgcoord = K @ extr_3x3
        worldgrid2imgcoord = worldcoord2imgcoord @ worldgrid2worldcoord
        imgcoord2worldgrid = inv(worldgrid2imgcoord)
        proj_mat = permutation @ imgcoord2worldgrid
    
    其中 permutation 用于交换 (u, v) 坐标顺序。
    
    Args:
        K_feat: 缩放后的相机内参 (3, 3)
                已根据特征图尺寸进行过缩放
        R: 旋转矩阵 (3, 3)，world to camera
        t: 平移向量 (3, 1)，world to camera
        worldgrid2worldcoord: 世界网格坐标到世界坐标的变换矩阵 (3, 3)
        
    Returns:
        np.ndarray: 投影矩阵 (3, 3)，dtype=float64
                   可用于将 BEV 网格点投影到图像特征平面
                   
    Raises:
        np.linalg.LinAlgError: 如果投影矩阵奇异（无法求逆）
        
    Example:
        >>> from calibration import read_intrinsics, read_opencv_xml_vec
        >>> K, dist = read_intrinsics(Path("intr.xml"))
        >>> rvec = read_opencv_xml_vec(Path("extr.xml"), "rvec")
        >>> tvec = read_opencv_xml_vec(Path("extr.xml"), "tvec")
        >>> R, _ = cv2.Rodrigues(rvec)
        >>> K_feat = scale_intrinsics(K, 0.25, 0.25)
        >>> w2w = make_worldgrid2worldcoord_mat(-3.0, -9.0, 0.1)
        >>> proj = build_mvdet_proj_mat(K_feat, R, tvec.reshape(3, 1), w2w)
        >>> print(proj.shape)
        (3, 3)
    """
    # 外参矩阵：world to camera (3x4)
    extr = np.concatenate([R, t.reshape(3, 1)], axis=1)
    
    # 移除 Z 列，得到 (3x3)：[R1 R2 t]
    # 这对应于 Hartley-Zisserman 的齐次坐标投影
    extr_3x3 = np.delete(extr, 2, 1)
    
    # 世界坐标到图像坐标的变换
    worldcoord2imgcoord = K_feat @ extr_3x3  # (3, 3)
    
    # 世界网格坐标到图像坐标的变换
    worldgrid2imgcoord = worldcoord2imgcoord @ worldgrid2worldcoord  # (3, 3)
    
    # 反向变换：图像坐标回到世界网格坐标
    imgcoord2worldgrid = np.linalg.inv(worldgrid2imgcoord)
    
    # 交换坐标顺序的排列矩阵
    # 从 (u, v, 1) 到 (v, u, 1)（或反之）
    # 这确保与 MVDet 源码的坐标约定一致
    permutation = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    proj = permutation @ imgcoord2worldgrid
    
    return proj


def compute_valid_ratio_from_homography(
    M_src2dst: np.ndarray,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int]
) -> float:
    """
    用 homography 矩阵估计投影的有效比例
    
    当将特征图投影到 BEV 网格时，并非所有 BEV 点都映射
    到有效的图像区域。此函数估计有效映射的比例。
    
    方法：
    1. 获取 BEV 网格的所有点（逆向映射回图像）
    2. 计算这些点中有多少落在图像有效范围内
    3. 返回有效点的比例
    
    Args:
        M_src2dst: 从源到目标的 homography 矩阵 (3, 3)
                   src 是图像特征平面，dst 是 BEV 网格
        src_hw: 源（图像特征平面）的 (height, width)
        dst_hw: 目标（BEV 网格）的 (height, width)
        
    Returns:
        float: 有效投影比例，范围 [0.0, 1.0]
               - 1.0 表示完全覆盖
               - 0.5 表示 50% 覆盖
               - 0.0 表示无覆盖
               
    Example:
        >>> M = np.eye(3)
        >>> ratio = compute_valid_ratio_from_homography(M, (270, 480), (135, 240))
        >>> print(f"Valid ratio: {ratio:.4f}")
    """
    Hs, Ws = src_hw
    Hd, Wd = dst_hw
    
    try:
        M_inv = np.linalg.inv(M_src2dst)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，无法求逆，返回 0
        return 0.0
    
    # 生成 BEV 网格坐标
    xs = np.arange(Wd, dtype=np.float64)
    ys = np.arange(Hd, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (Hd, Wd)
    
    # 齐次坐标
    ones = np.ones_like(xx)
    dst = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T  # (3, Hd*Wd)
    
    # 逆向投影回图像特征平面
    src = M_inv @ dst
    x = src[0] / (src[2] + 1e-9)
    y = src[1] / (src[2] + 1e-9)
    
    # 检查是否在有效范围内
    valid = (x >= 0) & (x <= (Ws - 1)) & (y >= 0) & (y <= (Hs - 1))
    
    return float(valid.mean())


def warp_perspective_torch(
    src: torch.Tensor,
    M_src2dst: torch.Tensor,
    dsize: Tuple[int, int]
) -> torch.Tensor:
    """
    PyTorch 纯实现的透视变换
    
    将源图像按照 homography 矩阵投影到目标空间。
    对标 OpenCV 的 warp_perspective 和 Kornia 的实现。
    
    计算流程：
    1. 生成目标网格中的所有点
    2. 通过逆 homography 矩阵映射回源坐标
    3. 使用双线性插值采样源图像
    4. 填充越界区域
    
    Args:
        src: 源特征图 (B, C, Hs, Ws)
             B: 批大小，C: 通道数
        M_src2dst: Homography 矩阵 (B, 3, 3) 或 (3, 3)
                   定义从源到目标的变换：dst_coord = M @ src_coord
        dsize: 目标尺寸 (Hd, Wd)
        
    Returns:
        torch.Tensor: 投影后的特征图 (B, C, Hd, Wd)
        
    Shape:
        - src: (B, C, Hs, Ws)
        - M_src2dst: (B, 3, 3) or (3, 3)
        - dsize: (Hd, Wd)
        - output: (B, C, Hd, Wd)
        
    Note:
        - 采样模式：双线性插值
        - 填充模式：零填充（越界返回 0）
        - 对齐方式：align_corners=True（对齐网格角点）
        
    Example:
        >>> src = torch.randn(2, 512, 135, 240)
        >>> M = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        >>> dst = warp_perspective_torch(src, M, (135, 240))
        >>> print(dst.shape)
        torch.Size([2, 512, 135, 240])
    """
    B, C, Hs, Ws = src.shape
    Hd, Wd = dsize
    device = src.device
    dtype = src.dtype
    
    # 确保 M 是 (B, 3, 3)
    if M_src2dst.dim() == 2:
        M_src2dst = M_src2dst.unsqueeze(0).expand(B, -1, -1)
    
    # 生成目标网格中的像素坐标
    ys, xs = torch.meshgrid(
        torch.arange(Hd, device=device, dtype=dtype),
        torch.arange(Wd, device=device, dtype=dtype),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    
    # 齐次坐标：(3, Hd*Wd)
    dst_h = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # (3, Hd*Wd)
    dst_h = dst_h.unsqueeze(0).expand(B, -1, -1)  # (B, 3, Hd*Wd)
    
    # 逆向投影：src_h = inv(M) @ dst_h
    M_inv = torch.inverse(M_src2dst.to(dtype=dtype, device=device))  # (B, 3, 3)
    src_h = M_inv @ dst_h  # (B, 3, Hd*Wd)
    
    # 从齐次坐标恢复笛卡尔坐标
    x = src_h[:, 0] / src_h[:, 2].clamp_min(1e-6)
    y = src_h[:, 1] / src_h[:, 2].clamp_min(1e-6)
    
    # 归一化到 [-1, 1] 用于 grid_sample
    # grid_sample 期望在 [-1, 1] 范围内，其中 (-1, -1) 是左上角，(1, 1) 是右下角
    x_norm = 2.0 * (x / (Ws - 1.0)) - 1.0
    y_norm = 2.0 * (y / (Hs - 1.0)) - 1.0
    
    # 构建采样网格：(B, Hd, Wd, 2)
    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(B, Hd, Wd, 2)
    
    # 使用 grid_sample 进行双线性插值
    # padding_mode="zeros" 表示越界返回 0
    # align_corners=True 对齐网格角点
    out = F.grid_sample(
        src, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )
    
    return out


def create_grid_sampler(
    proj_mat: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    device: torch.device
) -> callable:
    """
    创建网格采样函数
    
    用于多次采样相同配置的数据时提高效率。
    预计算投影矩阵及其逆，避免重复计算。
    
    Args:
        proj_mat: 投影矩阵 (3, 3) 或 (B, 3, 3)
        src_hw: 源尺寸 (Hs, Ws)
        dst_hw: 目标尺寸 (Hd, Wd)
        device: 计算设备
        
    Returns:
        callable: 采样函数 f(src_feature) -> dst_feature
        
    Example:
        >>> proj_mat = torch.eye(3)
        >>> sampler = create_grid_sampler(proj_mat, (270, 480), (135, 240), torch.device("cuda"))
        >>> feature = torch.randn(2, 512, 270, 480).cuda()
        >>> output = sampler(feature)
        >>> print(output.shape)
        torch.Size([2, 512, 135, 240])
    """
    Hs, Ws = src_hw
    Hd, Wd = dst_hw
    
    # 预计算网格
    ys, xs = torch.meshgrid(
        torch.arange(Hd, device=device, dtype=torch.float32),
        torch.arange(Wd, device=device, dtype=torch.float32),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    dst_h = torch.stack([xs, ys, ones], dim=0).reshape(3, -1).unsqueeze(0)
    
    def sampler(src: torch.Tensor) -> torch.Tensor:
        B = src.shape[0]
        dtype = src.dtype
        
        M = proj_mat.to(dtype=dtype, device=device)
        if M.dim() == 2:
            M = M.unsqueeze(0).expand(B, -1, -1)
        
        M_inv = torch.inverse(M)
        src_h = M_inv @ dst_h.to(dtype=dtype).expand(B, -1, -1)
        
        x = src_h[:, 0] / src_h[:, 2].clamp_min(1e-6)
        y = src_h[:, 1] / src_h[:, 2].clamp_min(1e-6)
        
        x_norm = 2.0 * (x / (Ws - 1.0)) - 1.0
        y_norm = 2.0 * (y / (Hs - 1.0)) - 1.0
        
        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(B, Hd, Wd, 2)
        
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    
    return sampler
