# scripts/calibration.py
"""
相机标定模块：读取和处理相机标定参数
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import cv2

from config import UNIT_SCALE_THRESHOLD_STEP, UNIT_SCALE_THRESHOLD_TVEC, UNIT_SCALE_CM, UNIT_SCALE_M


def parse_rectangles_pom(pom_path: Path) -> Dict[str, float]:
    """
    解析 rectangles.pom 文件
    
    该文件包含 Wildtrack 数据集的固定参数：
    - NB_WIDTH, NB_HEIGHT: 网格分辨率
    - ORIGINE_X, ORIGINE_Y: 世界坐标系原点
    - STEP: 网格步长
    
    Args:
        pom_path: rectangles.pom 文件路径
        
    Returns:
        dict: 键值对字典，所有值为 float
        如果文件不存在，返回空字典
        
    Example:
        >>> pom_dict = parse_rectangles_pom(Path("wildtrack/rectangles.pom"))
        >>> print(pom_dict.get("NB_WIDTH", 480))
        480
    """
    if not pom_path.exists():
        return {}
    
    txt = pom_path.read_text(encoding="utf-8", errors="ignore")
    kv = {}
    
    for line in txt.splitlines():
        line = line.strip()
        
        # 跳过空行和注释
        if not line or line.startswith("#"):
            continue
        
        # 解析 key=value 格式
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                kv[k] = float(v)
            except (ValueError, TypeError):
                # 跳过无法转换的值
                pass
    
    return kv


def read_intrinsics(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 OpenCV XML 文件读取相机内参
    
    Args:
        xml_path: XML 标定文件路径
        
    Returns:
        tuple: (K, dist)
            - K: 相机内参矩阵 (3, 3)，float64
            - dist: 畸变系数，1D 数组，float64
            
    Raises:
        cv2.error: 如果 XML 格式无效
        FileNotFoundError: 如果文件不存在
        
    Example:
        >>> K, dist = read_intrinsics(Path("calib/intr_CVLab1.xml"))
        >>> print(K.shape, dist.shape)
        (3, 3) (5,)
    """
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    
    fs.release()
    
    K = np.array(K, dtype=np.float64)
    dist = np.array(dist, dtype=np.float64).reshape(-1)
    
    return K, dist


def read_opencv_xml_vec(xml_path: Path, key: str) -> np.ndarray:
    """
    从 OpenCV XML 文件读取向量
    
    用于读取旋转向量 (rvec) 和平移向量 (tvec)
    
    Args:
        xml_path: XML 标定文件路径
        key: 键名，例如 "rvec" 或 "tvec"
        
    Returns:
        np.ndarray: 向量，float64，形状为 (N,)
        
    Example:
        >>> rvec = read_opencv_xml_vec(Path("calib/extr_CVLab1.xml"), "rvec")
        >>> tvec = read_opencv_xml_vec(Path("calib/extr_CVLab1.xml"), "tvec")
        >>> print(rvec.shape, tvec.shape)
        (3,) (3,)
    """
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    node = fs.getNode(key)
    
    # 逐个读取向量元素
    vals = [node.at(i).real() for i in range(node.size())]
    
    fs.release()
    
    return np.array(vals, dtype=np.float64)


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    缩放相机内参矩阵
    
    当图像尺寸改变时需要缩放内参中的焦距和主点坐标。
    
    Args:
        K: 原始内参矩阵 (3, 3)
        sx: x 方向缩放因子（新宽度 / 原宽度）
        sy: y 方向缩放因子（新高度 / 原高度）
        
    Returns:
        np.ndarray: 缩放后的内参矩阵 (3, 3)
        
    Formula:
        K_new[0, 0] = K_old[0, 0] * sx  (焦距 fx)
        K_new[1, 1] = K_old[1, 1] * sy  (焦距 fy)
        K_new[0, 2] = K_old[0, 2] * sx  (主点 cx)
        K_new[1, 2] = K_old[1, 2] * sy  (主点 cy)
        
    Example:
        >>> K = np.array([[1920, 0, 960], [0, 1920, 540], [0, 0, 1]], dtype=np.float64)
        >>> K_scaled = scale_intrinsics(K, sx=0.5, sy=0.5)
        >>> print(K_scaled[0, 0], K_scaled[1, 1], K_scaled[0, 2])
        960.0 960.0 480.0
    """
    K2 = K.copy()
    K2[0, 0] *= sx    # 焦距 fx
    K2[1, 1] *= sy    # 焦距 fy
    K2[0, 2] *= sx    # 主点 cx
    K2[1, 2] *= sy    # 主点 cy
    return K2


def decide_unit_scale(step_m: float, t_norms: list) -> float:
    """
    自动推断标定数据的单位制
    
    启发式规则：
    - 如果网格步长 < 1.0 且 median(||tvec||) > 50.0
      => 假设标定使用厘米，unit_scale = 100
    - 否则 => 使用米为单位，unit_scale = 1
    
    这是因为：
    - 如果步长是 0.025m，网格尺寸合理
    - 但如果 tvec 的绝对值在 100-1000 范围内
    - 则很可能 tvec 的单位是厘米而不是米
    
    Args:
        step_m: 网格步长（以米为单位）
        t_norms: 相机平移向量的范数列表，shape (num_views,)
        
    Returns:
        float: 单位转换因子
            - 100.0 表示使用厘米
            - 1.0 表示使用米
            
    Example:
        >>> unit_scale = decide_unit_scale(0.025, [150, 200, 180])
        >>> print(unit_scale)
        100.0
    """
    unit_scale = UNIT_SCALE_M
    
    if (step_m < UNIT_SCALE_THRESHOLD_STEP and
        np.median(t_norms) > UNIT_SCALE_THRESHOLD_TVEC):
        unit_scale = UNIT_SCALE_CM
    
    return unit_scale


class CalibrationLoader:
    """
    标定数据加载器
    
    统一管理相机标定参数的读取和缓存。
    
    Attributes:
        calib_root: 标定文件根目录
        cam_names: 相机名称列表
        cache: 标定数据缓存
    """
    
    def __init__(self, calib_root: Path, cam_names: list):
        """
        初始化标定加载器
        
        Args:
            calib_root: 标定文件根目录，应包含：
                - intrinsic_original/intr_*.xml
                - extrinsic/extr_*.xml
            cam_names: 相机名称列表
        """
        self.calib_root = calib_root
        self.cam_names = cam_names
        self.cache: Dict[int, Dict[str, Any]] = {}
    
    def load(self, view_id: int) -> Dict[str, Any]:
        """
        加载单个相机的标定参数
        
        Args:
            view_id: 视角 ID（0 到 num_cams-1）
            
        Returns:
            dict: 标定数据，包含：
                - "K0": 内参矩阵 (3, 3)
                - "R": 旋转矩阵 (3, 3)
                - "t": 平移向量 (3, 1)
                - "dist": 畸变系数
                - "cam": 相机名称
                
        Raises:
            FileNotFoundError: 如果标定文件不存在
            IndexError: 如果 view_id 超出范围
        """
        if view_id in self.cache:
            return self.cache[view_id]
        
        cam_name = self.cam_names[view_id]
        intr_path = self.calib_root / "intrinsic_original" / f"intr_{cam_name}.xml"
        extr_path = self.calib_root / "extrinsic" / f"extr_{cam_name}.xml"
        
        if not intr_path.exists():
            raise FileNotFoundError(f"内参文件不存在: {intr_path}")
        if not extr_path.exists():
            raise FileNotFoundError(f"外参文件不存在: {extr_path}")
        
        # 读取内参
        K0, dist = read_intrinsics(intr_path)
        
        # 读取外参
        rvec = read_opencv_xml_vec(extr_path, "rvec").reshape(3, 1)
        tvec = read_opencv_xml_vec(extr_path, "tvec").reshape(3, 1)
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        t = tvec.astype(np.float64)
        
        calib_data = {
            "K0": K0,
            "R": R,
            "t": t,
            "dist": dist,
            "cam": cam_name
        }
        
        self.cache[view_id] = calib_data
        return calib_data
    
    def load_all(self, view_ids: list) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray]:
        """
        加载多个相机的标定参数
        
        Args:
            view_ids: 视角 ID 列表
            
        Returns:
            tuple: (calib_cache, t_norms)
                - calib_cache: 标定数据字典
                - t_norms: 平移向量范数数组
                
        Example:
            >>> loader = CalibrationLoader(Path("calib"), ["CVLab1", "CVLab2"])
            >>> calib_cache, t_norms = loader.load_all([0, 1])
            >>> print(t_norms)
            array([150.5, 200.3])
        """
        calib_cache = {}
        t_norms = []
        
        for v in view_ids:
            calib = self.load(v)
            calib_cache[v] = calib
            t_norm = float(np.linalg.norm(calib["t"]))
            t_norms.append(t_norm)
        
        return calib_cache, np.array(t_norms)
