# scripts/config.py
"""
配置文件：包含所有常量和默认参数
对应 Wildtrack 数据集的固定参数及网络超参数
"""

# ============================================================================
# Wildtrack 数据集坐标系常量（来自 rectangles.pom）
# ============================================================================

# 世界坐标系原点（米）
ORIGINE_X_M = -3.0
ORIGINE_Y_M = -9.0

# BEV 网格分辨率
NB_WIDTH = 480      # x 方向网格数（单位：0.025m）
NB_HEIGHT = 1440    # y 方向网格数

# 网格步长计算（米）
WIDTH_M = 12.0                        # 实际宽度（米）
STEP_M = WIDTH_M / NB_WIDTH           # 0.025 m/bin


# ============================================================================
# 原始图像参数
# ============================================================================

IMG_ORI_W = 1920    # 原始图像宽度（像素）
IMG_ORI_H = 1080    # 原始图像高度（像素）


# ============================================================================
# 相机名称映射
# ============================================================================

CAM_NAMES = [
    "CVLab1", "CVLab2", "CVLab3", "CVLab4",
    "IDIAP1", "IDIAP2", "IDIAP3"
]


# ============================================================================
# 默认训练参数
# ============================================================================

# 数据相关
DEFAULT_MAX_FRAMES = 300
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 2

# 网络架构
DEFAULT_BEV_DOWN = 4                   # BEV 下采样倍数
DEFAULT_FEAT_H = 270                   # 特征图高度
DEFAULT_FEAT_W = 480                   # 特征图宽度
DEFAULT_IMG_H = 720                    # 输入图像高度
DEFAULT_IMG_W = 1280                   # 输入图像宽度
DEFAULT_FEAT_CH = 512                  # 特征通道数

# 人体模型
DEFAULT_PERSON_H = 1.7                 # 人体高度（米）

# 损失函数
DEFAULT_ALPHA = 1.0                    # 图像损失权重
DEFAULT_MAP_KSIZE = 11                 # BEV 热图高斯核大小
DEFAULT_MAP_SIGMA = 2.5                # BEV 热图高斯标准差
DEFAULT_IMG_KSIZE = 11                 # 图像热图高斯核大小
DEFAULT_IMG_SIGMA = 2.0                # 图像热图高斯标准差

# 优化器
DEFAULT_MAX_LR = 0.1
DEFAULT_LR_INIT = 1e-3
DEFAULT_MOMENTUM = 0.5
DEFAULT_WEIGHT_DECAY = 5e-4

# 训练策略
DEFAULT_EPOCHS = 10
DEFAULT_LOG_EVERY = 20
DEFAULT_SAVE_STEPS = "0,200,1000"
DEFAULT_FIXED_STEM = "00000055"

# 视图过滤
DEFAULT_VALID_THR = 0.10               # 有效投影比例阈值

# 计算设备
DEFAULT_DEVICE = "cuda"
DEFAULT_AMP_ENABLED = False
DEFAULT_FREEZE_BN = False
DEFAULT_PRETRAINED = False

# 数据目录
DEFAULT_DATA_ROOT = "wildtrack"
DEFAULT_OUTPUT_DIR = "outputs/train_multicam_mvdet_style_v3"


# ============================================================================
# ImageNet 标准化参数
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# 单位转换规则
# ============================================================================

# 用于自动推断标定数据的单位制：
# 如果 step_m < 1.0 且 median(t_norms) > 50.0
# 则假设标定使用厘米，unit_scale = 100
# 否则 unit_scale = 1（保持米为单位）

UNIT_SCALE_THRESHOLD_STEP = 1.0
UNIT_SCALE_THRESHOLD_TVEC = 50.0
UNIT_SCALE_CM = 100.0
UNIT_SCALE_M = 1.0
