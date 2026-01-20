# scripts/dataset.py
"""
数据集模块：Wildtrack 多视角数据加载和标签生成
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import (
    NB_WIDTH, NB_HEIGHT,
    STEP_M, ORIGINE_X_M, ORIGINE_Y_M,
    IMAGENET_MEAN, IMAGENET_STD
)


class WildtrackMVDetDataset(Dataset):
    """
    Wildtrack 多视角检测数据集
    
    从 Wildtrack 数据集加载多视角图像和 3D 标签，
    生成 BEV 热图和单视角热图。
    
    数据流程：
    1. 加载多视角 RGB 图像
    2. 从注释中读取 3D 目标位置（网格索引）
    3. 将 3D 位置投影到各视角图像
    4. 生成 BEV 全局热图和单视角特征平面热图
    
    输出：
    - x_views: (V, 3, Hi, Wi) - 多视角 RGB 图像
    - map_gt: (1, NB_HEIGHT, NB_WIDTH) - BEV 全局热图
    - imgs_gt: (V, 2, Hf, Wf) - 单视角特征平面热图（head/foot）
    
    Attributes:
        data_root: 数据集根目录
        views: 使用的视角 ID 列表
        ann_files: 注释文件路径列表
        img_dirs: 各视角的图像目录列表
        calib_cache: 标定数据缓存
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
        """
        初始化数据集
        
        Args:
            data_root: Wildtrack 数据集根目录
            views: 使用的视角 ID 列表，例如 [0, 1, 2]
            max_frames: 最大帧数，-1 表示使用全部
            img_hw: 输入图像大小 (height, width)
            feat_hw: 特征平面大小 (height, width)
            bev_down: BEV 下采样倍数
            person_h_m: 人体高度（米或厘米，单位与标定一致）
            unit_scale: 单位转换因子（1.0 或 100.0）
            calib_cache: 标定数据缓存（包含内参、外参）
        """
        self.data_root = Path(data_root)
        self.views = views
        self.Hi, self.Wi = img_hw
        self.Hf, self.Wf = feat_hw
        self.bev_down = bev_down
        self.person_h = person_h_m * unit_scale
        self.unit_scale = unit_scale
        self.calib_cache = calib_cache
        
        # 注释文件目录
        self.ann_dir = self.data_root / "annotations_positions"
        assert self.ann_dir.exists(), f"注释目录不存在: {self.ann_dir}"
        
        # 各视角的图像目录
        self.img_dirs = []
        for v in views:
            p = self.data_root / "Image_subsets" / f"C{v+1}"
            assert p.exists(), f"图像目录不存在: {p}"
            self.img_dirs.append(p)
        
        # 收集所有注释文件
        self.ann_files = sorted(self.ann_dir.glob("*.json"))
        if max_frames > 0:
            self.ann_files = self.ann_files[:max_frames]
        assert len(self.ann_files) > 0, "没有找到注释文件"
        
        # ImageNet 标准化参数
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        
        # BEV 网格参数
        self.nb_w = NB_WIDTH
        self.nb_h = NB_HEIGHT
        
        # 世界坐标系参数（已根据 unit_scale 缩放）
        step = STEP_M * unit_scale
        self.ox = ORIGINE_X_M * unit_scale
        self.oy = ORIGINE_Y_M * unit_scale
        self.step = step
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.ann_files)
    
    def _load_image(self, img_dir: Path, stem: str) -> Image.Image:
        """
        从目录加载图像
        
        尝试多个常见图像格式
        
        Args:
            img_dir: 图像目录
            stem: 文件名前缀（不含扩展名）
            
        Returns:
            Image.Image: RGB 图像对象
            
        Raises:
            FileNotFoundError: 如果找不到任何格式的图像
        """
        for ext in [".png", ".jpg", ".jpeg"]:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"找不到图像: {stem} in {img_dir}")
    
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple:
                - stem: 样本名称（文件名前缀）
                - x_views: 多视角图像 (V, 3, Hi, Wi)
                - map_gt: BEV 全局热图 (1, NB_HEIGHT, NB_WIDTH)
                - imgs_gt: 单视角特征平面热图 (V, 2, Hf, Wf)
        """
        ann_path = self.ann_files[idx]
        stem = ann_path.stem
        
        # 加载注释
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        
        # 加载多视角图像
        xs = []
        for img_dir in self.img_dirs:
            img = self._load_image(img_dir, stem).resize((self.Wi, self.Hi), Image.BILINEAR)
            x = torch.from_numpy(np.array(img, dtype=np.uint8)).float() / 255.0
            x = x.permute(2, 0, 1)  # (3, Hi, Wi)
            x = (x - self.mean) / self.std  # 标准化
            xs.append(x)
        x_views = torch.stack(xs, dim=0)  # (V, 3, Hi, Wi)
        
        # 初始化标签
        map_gt = torch.zeros((1, self.nb_h, self.nb_w), dtype=torch.float32)
        imgs_gt = torch.zeros((len(self.views), 2, self.Hf, self.Wf), dtype=torch.float32)
        
        # 处理每个目标
        for obj in data:
            pos_id = obj.get("positionID", None)
            if pos_id is None:
                continue
            
            pos_id = int(pos_id)
            ix = pos_id % self.nb_w  # x 网格索引
            iy = pos_id // self.nb_w  # y 网格索引
            
            # 生成 BEV 全局热图
            if 0 <= iy < self.nb_h and 0 <= ix < self.nb_w:
                map_gt[0, iy, ix] = 1.0
            
            # 转换到世界坐标系（使用网格中心）
            Xw = self.ox + (ix + 0.5) * self.step
            Yw = self.oy + (iy + 0.5) * self.step
            
            # 创建脚部和头部的 3D 点
            Pw_foot = np.array([Xw, Yw, 0.0], dtype=np.float64).reshape(3, 1)
            Pw_head = np.array([Xw, Yw, self.person_h], dtype=np.float64).reshape(3, 1)
            
            # 投影到各视角的特征平面
            for vi, v in enumerate(self.views):
                calib = self.calib_cache[v]
                Kf = calib["K_feat"]  # 缩放后的内参
                R = calib["R"]        # 旋转矩阵
                t = calib["t"]        # 平移向量
                
                # 投影函数（忽略畸变，对齐 MVDet 假设）
                def proj(Pw):
                    # 世界坐标 -> 相机坐标
                    Pc = R @ Pw + t
                    z = float(Pc[2, 0])
                    if z <= 1e-6:
                        return None
                    # 相机坐标 -> 图像坐标
                    u = (Kf[0, 0] * (Pc[0, 0] / z) + Kf[0, 2])
                    v_ = (Kf[1, 1] * (Pc[1, 0] / z) + Kf[1, 2])
                    return float(u), float(v_)
                
                # 投影头部
                p_head = proj(Pw_head)
                if p_head is not None:
                    u, v_ = p_head
                    x = int(round(u))
                    y = int(round(v_))
                    if 0 <= x < self.Wf and 0 <= y < self.Hf:
                        imgs_gt[vi, 0, y, x] = 1.0
                
                # 投影脚部
                p_foot = proj(Pw_foot)
                if p_foot is not None:
                    u, v_ = p_foot
                    x = int(round(u))
                    y = int(round(v_))
                    if 0 <= x < self.Wf and 0 <= y < self.Hf:
                        imgs_gt[vi, 1, y, x] = 1.0
        
        return stem, x_views, map_gt, imgs_gt


def create_wildtrack_dataset(
    data_root: Path,
    views: List[int],
    max_frames: int,
    img_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
    bev_down: int,
    person_h_m: float,
    unit_scale: float,
    calib_cache: Dict[int, Dict[str, Any]],
) -> WildtrackMVDetDataset:
    """
    工厂函数：创建 Wildtrack 数据集
    
    Args:
        data_root: 数据集根目录
        views: 视角 ID 列表
        max_frames: 最大帧数
        img_hw: 输入图像大小
        feat_hw: 特征平面大小
        bev_down: BEV 下采样倍数
        person_h_m: 人体高度
        unit_scale: 单位转换因子
        calib_cache: 标定缓存
        
    Returns:
        WildtrackMVDetDataset: 初始化的数据集
    """
    return WildtrackMVDetDataset(
        data_root=data_root,
        views=views,
        max_frames=max_frames,
        img_hw=img_hw,
        feat_hw=feat_hw,
        bev_down=bev_down,
        person_h_m=person_h_m,
        unit_scale=unit_scale,
        calib_cache=calib_cache,
    )
