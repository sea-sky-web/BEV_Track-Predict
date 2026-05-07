# BEV_Track&Predict Code Wiki

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [模块职责详解](#3-模块职责详解)
   - [config.py](#31-configpy)
   - [calibration.py](#32-calibrationpy)
   - [geometry.py](#33-geometrypy)
   - [dataset.py](#34-datasetpy)
   - [models.py](#35-modelspy)
   - [loss.py](#36-losspy)
   - [trainer.py](#37-trainerpy)
   - [utils.py](#38-utilspy)
   - [train_main.py](#39-train_mainpy)
   - [evaluate_main.py](#310-evaluate_mainpy)
4. [依赖关系图](#4-依赖关系图)
5. [数据流与执行流程](#5-数据流与执行流程)
6. [关键类与函数说明](#6-关键类与函数说明)
7. [项目运行方式](#7-项目运行方式)
8. [数据集要求](#8-数据集要求)
9. [常用配置参数](#9-常用配置参数)

---

## 1. 项目概述

**项目名称**: BEV_Track&Predict  
**目标**: 在 Wildtrack 数据集上快速搭建并跑通一个"多视角→BEV→行人占据/定位(POM)→后续关联与预测"的可训练闭环原型。

**核心能力**:
- 多视角图像特征提取
- 透视变换投影到BEV空间
- BEV热图预测（行人位置）
- 单视角辅助监督（人头/人脚热图）
- 检测级评估（Precision/Recall/F1/定位误差）

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练流程                                     │
├─────────────────────────────────────────────────────────────────────┤
│  数据输入 (多视角RGB)                                               │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ calibration │───▶│  geometry   │───▶│  dataset    │             │
│  │  (标定加载)  │    │ (投影矩阵)   │    │ (数据加载)  │             │
│  └─────────────┘    └─────────────┘    └──────┬──────┘             │
│                                                │                    │
│                                                ▼                    │
│                                      ┌──────────────────┐          │
│                                      │    DataLoader    │          │
│                                      └────────┬─────────┘          │
│                                               │                     │
│                                               ▼                     │
│                           ┌───────────────────────────────┐         │
│                           │         models.py              │         │
│                           │  ┌─────────────────────────┐   │         │
│                           │  │ ResNet50 Backbone      │   │         │
│                           │  │ (共享特征提取)          │   │         │
│                           │  └───────────┬───────────┘   │         │
│                           │              │                 │         │
│                           │    ┌─────────┴─────────┐      │         │
│                           │    ▼                   ▼      │         │
│                           │  ImgHeadFoot      BEVHead     │         │
│                           │  (单视角辅助)    (BEV融合)    │         │
│                           │    │                   │      │         │
│                           │    └─────────┬─────────┘      │         │
│                           │              ▼                 │         │
│                           │     输出: BEV热图 + 图像热图    │         │
│                           └───────────────────────────────┘         │
│                                               │                     │
│                                               ▼                     │
│                           ┌───────────────────────────────┐         │
│                           │         trainer.py            │         │
│                           │  - GaussianMSE损失计算        │         │
│                           │  - 优化器/Scheduler           │         │
│                           │  - 检查点管理                  │         │
│                           └───────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 模块职责详解

### 3.1 config.py

**职责**: 存储所有常量和默认超参数。

**核心内容**:
- **数据集常量**: 世界坐标系原点、BEV网格分辨率、步长
- **图像参数**: 原始图像尺寸、特征图尺寸、输入图像尺寸
- **相机名称**: 7个相机的名称映射
- **训练参数**: batch大小、epoch数、学习率、动量、权重衰减
- **损失参数**: 高斯核大小和标准差
- **单位转换规则**: 自动推断标定数据单位（米/厘米）

**关键常量**:
```python
# 世界坐标系（米）
ORIGINE_X_M = -3.0
ORIGINE_Y_M = -9.0
NB_WIDTH = 480      # BEV网格宽度（像素）
NB_HEIGHT = 1440    # BEV网格高度（像素）
STEP_M = 0.025      # 网格步长（米）

# 默认训练参数
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR_INIT = 1e-3
DEFAULT_MAX_LR = 0.1
```

---

### 3.2 calibration.py

**职责**: 相机标定数据的读取、解析和单位制推断。

**核心类/函数**:

| 类/函数 | 作用 | 关键输入输出 |
|---------|------|-------------|
| `CalibrationLoader` | 标定数据加载器 | 加载内参(K)、外参(R,t) |
| `parse_rectangles_pom()` | 解析pom文件 | 读取网格参数 |
| `read_intrinsics()` | 读取内参XML | 返回K矩阵和畸变系数 |
| `scale_intrinsics()` | 缩放内参矩阵 | 根据特征图尺寸调整K |
| `decide_unit_scale()` | 自动推断单位制 | 根据t向量范数判断米/厘米 |

**标定文件结构**:
```
wildtrack/calibrations/
├── intrinsic_zero/intr_<CAM>.xml   # 内参
└── extrinsic/extr_<CAM>.xml        # 外参(rvec, tvec)
```

---

### 3.3 geometry.py

**职责**: 几何变换和投影矩阵构建，实现MVDet风格的透视变换。

**核心函数**:

| 函数 | 作用 | 数学原理 |
|------|------|---------|
| `make_worldgrid2worldcoord_mat()` | 构建网格→世界坐标变换矩阵 | X = origin_x + (grid_x+0.5)*step |
| `build_mvdet_proj_mat()` | 构建MVDet投影矩阵 | K·[R|t][:, :3]·worldgrid2worldcoord |
| `compute_valid_ratio_from_homography()` | 计算投影有效比例 | 逆向投影统计有效点比例 |
| `warp_perspective_torch()` | PyTorch透视变换 | 使用grid_sample实现双线性插值 |

**投影变换链**:
```
世界网格坐标 → 世界坐标系 → 相机坐标系 → 图像坐标系 → 特征平面
```

---

### 3.4 dataset.py

**职责**: Wildtrack多视角数据集加载和标签生成。

**核心类**: `WildtrackMVDetDataset`

**数据流程**:
1. **加载多视角图像** → 标准化（ImageNet均值/std）
2. **读取3D标签** → positionID映射到BEV网格坐标
3. **生成BEV热图** → 目标位置设置为1.0
4. **投影到各视角** → 生成单视角head/foot热图

**输出格式**:
```python
# stem: 样本名称
# x_views: (V, 3, Hi, Wi) - 多视角图像
# map_gt: (1, NB_HEIGHT, NB_WIDTH) - BEV热图
# imgs_gt: (V, 2, Hf, Wf) - 单视角head/foot热图
```

---

### 3.5 models.py

**职责**: MVDet风格的多视角BEV检测网络架构。

**核心组件**:

| 组件 | 作用 | 关键设计 |
|------|------|---------|
| `ResNet50Stride8Trunk` | 共享特征提取主干 | 使用空洞卷积保持stride=8 |
| `ImgHeadFoot` | 单视角辅助预测头 | 输出2通道（head/foot） |
| `BEVHeadDilated` | BEV融合预测头 | 递增dilation扩大感受野 |
| `MVDetLikeNet` | 完整网络 | 多视角→BEV端到端 |

**网络架构**:
```
输入: (B, V, 3, Hi, Wi)
        │
        ▼
    [ResNet50主干] ← 共享权重
        │
        ▼
    [ImgHeadFoot] ← 辅助监督
        │
        ▼
    [透视变换] ← proj_mats
        │
        ▼
    [BEV特征拼接 + 坐标编码]
        │
        ▼
    [BEVHeadDilated]
        │
        ▼
输出: (map_logits, imgs_logits)
```

---

### 3.6 loss.py

**职责**: 实现MVDet风格的GaussianMSE损失函数。

**核心类**:

| 类 | 作用 | 特点 |
|----|------|-----|
| `GaussianMSE` | 高斯MSE损失（主损失） | 对GT进行高斯模糊，提供软标签 |
| `WeightedGaussianMSE` | 带权重的高斯MSE | 支持正负样本加权 |

**损失计算流程**:
1. 自适应max池化GT到预测尺寸
2. 应用高斯卷积平滑GT（软标签）
3. 计算预测与平滑GT的MSE

---

### 3.7 trainer.py

**职责**: 训练循环、损失计算、日志记录和检查点管理。

**核心类**: `MVDetTrainer`

**主要方法**:
- `train_epoch()`: 训练一个epoch，包含前向/反向传播
- `validate()`: 验证评估
- `save_checkpoint()`: 保存模型检查点
- `save_visualizations()`: 保存热图可视化

**训练配置**:
- 优化器: SGD（momentum=0.5）
- 学习率调度: OneCycleLR
- AMP支持: 自动混合精度训练

---

### 3.8 utils.py

**职责**: 通用工具函数。

**核心函数**:

| 函数 | 作用 |
|------|------|
| `save_heat_png()` | 将热图数组保存为PNG图像 |
| `build_gaussian_kernel_2d()` | 构建二维高斯卷积核 |
| `apply_gaussian_blur()` | 对热图应用高斯模糊 |

---

### 3.9 train_main.py

**职责**: 训练入口脚本，负责参数解析和组件初始化。

**执行流程**:
1. **解析命令行参数**
2. **加载标定数据** → 推断单位制
3. **构建投影矩阵** → 过滤低有效性视角
4. **创建数据集** → DataLoader
5. **创建模型** → MVDetLikeNet
6. **创建优化器和调度器**
7. **创建训练器** → MVDetTrainer
8. **训练循环** → 保存检查点

---

### 3.10 evaluate_main.py

**职责**: 离线评估脚本，支持损失指标和检测级评估。

**评估能力**:
1. **损失指标**: loss/bev_loss/img_loss/pos_mse/aux_pos_mse
2. **检测指标**: Precision/Recall/F1/定位误差（通过阈值扫描）

**检测评估流程**:
1. 阈值扫描（默认0.05~0.50）
2. NMS提取预测点
3. 匹配GT点（距离阈值）
4. 计算Precision/Recall/F1

---

## 4. 依赖关系图

```
config.py
    ├─> calibration.py -> geometry.py
    ├─> dataset.py
    ├─> train_main.py
    └─> evaluate_main.py

utils.py -> loss.py
geometry.py -> models.py
loss.py + models.py -> trainer.py

dataset.py + trainer.py + geometry.py + calibration.py + models.py -> train_main.py/evaluate_main.py
```

**外部依赖**:
```python
torch>=2.0
torchvision>=0.15
numpy
opencv-python
Pillow
```

---

## 5. 数据流与执行流程

### 5.1 训练数据流

```
数据准备阶段:
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ rectangles.pom   │     │ calibrations/    │     │ annotations/     │
│ (网格参数)       │     │ (内外参XML)      │     │ (3D标签JSON)     │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌───────────────────────────────────────────────────────────────────┐
│                        train_main.py                              │
├───────────────────────────────────────────────────────────────────┤
│  parse_rectangles_pom()                                          │
│  CalibrationLoader.load_all()                                    │
│  decide_unit_scale()                                             │
│  make_worldgrid2worldcoord_mat()                                 │
│  build_mvdet_proj_mat()                                          │
│  compute_valid_ratio_from_homography()                           │
│  create_wildtrack_dataset()                                      │
│  create_model()                                                  │
│  create_optimizer() + create_scheduler()                         │
│  MVDetTrainer()                                                  │
│  trainer.train_epoch()                                          │
└───────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ outputs/model*.pth│
└──────────────────┘
```

### 5.2 单样本前向流程

```
输入: x_views (B, V, 3, Hi, Wi)
        │
        ▼ (逐视角处理)
    ResNet50Stride8Trunk
        │
        ├──▶ ImgHeadFoot ──▶ imgs_logits (B, V, 2, Hf, Wf)
        │
        ▼
    F.interpolate (到特征尺寸)
        │
        ▼
    warp_perspective_torch (投影到BEV)
        │
        ▼
    torch.cat (多视角拼接)
        │
        ▼ (可选)
    添加坐标编码
        │
        ▼
    BEVHeadDilated
        │
        ▼
输出: map_logits (B, 1, Hb, Wb)
```

---

## 6. 关键类与函数说明

### 6.1 核心类速查

| 类 | 文件 | 核心功能 |
|----|------|---------|
| `CalibrationLoader` | calibration.py | 标定数据加载和缓存 |
| `WildtrackMVDetDataset` | dataset.py | 多视角数据集 |
| `ResNet50Stride8Trunk` | models.py | ResNet50主干（stride=8） |
| `ImgHeadFoot` | models.py | 单视角head/foot预测头 |
| `BEVHeadDilated` | models.py | BEV融合预测头 |
| `MVDetLikeNet` | models.py | 完整网络架构 |
| `GaussianMSE` | loss.py | 高斯MSE损失 |
| `MVDetTrainer` | trainer.py | 训练器 |

### 6.2 核心函数速查

| 函数 | 文件 | 作用 |
|------|------|------|
| `make_worldgrid2worldcoord_mat()` | geometry.py | 网格→世界坐标变换 |
| `build_mvdet_proj_mat()` | geometry.py | 构建投影矩阵 |
| `warp_perspective_torch()` | geometry.py | PyTorch透视变换 |
| `create_model()` | models.py | 模型工厂函数 |
| `create_wildtrack_dataset()` | dataset.py | 数据集工厂函数 |
| `create_optimizer()` | trainer.py | 创建SGD优化器 |
| `create_scheduler()` | trainer.py | 创建OneCycleLR |
| `build_gaussian_kernel_2d()` | utils.py | 构建高斯核 |

---

## 7. 项目运行方式

### 7.1 训练命令

```bash
# 基本训练（推荐）
python src/train_main.py --data_root wildtrack --views 0,1,2 --device cuda

# 完整参数示例
python src/train_main.py \
    --data_root wildtrack \
    --views 0,1,2 \
    --device cuda \
    --epochs 20 \
    --batch 2 \
    --max_frames 500 \
    --bev_down 4 \
    --lr_init 1e-3 \
    --max_lr 0.1 \
    --momentum 0.5 \
    --weight_decay 5e-4 \
    --alpha 1.0 \
    --drop_bad_views \
    --valid_thr 0.15 \
    --amp \
    --log_every 10
```

### 7.2 评估命令

```bash
# 基本评估
python src/evaluate_main.py \
    --data_root wildtrack \
    --views 0,1,2 \
    --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth \
    --device cuda

# 检测级评估（阈值扫描）
python src/evaluate_main.py \
    --data_root wildtrack \
    --views 1,2 \
    --drop_bad_views \
    --valid_thr 0.15 \
    --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth \
    --device cuda \
    --frame_start 300 \
    --max_frames 100 \
    --report_detection \
    --metrics_out outputs/eval_metrics.json
```

### 7.3 常用参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | wildtrack | 数据集根目录 |
| `--views` | str | 0,1,2 | 使用的视角ID |
| `--device` | str | cuda | 计算设备 |
| `--epochs` | int | 10 | 训练轮数 |
| `--batch` | int | 1 | 批大小 |
| `--max_frames` | int | 300 | 最大帧数 |
| `--bev_down` | int | 4 | BEV下采样倍数 |
| `--lr_init` | float | 1e-3 | 初始学习率 |
| `--max_lr` | float | 0.1 | 最大学习率 |
| `--alpha` | float | 1.0 | 图像损失权重 |
| `--drop_bad_views` | flag | False | 丢弃低有效性视角 |
| `--valid_thr` | float | 0.10 | 投影有效比例阈值 |
| `--amp` | flag | False | 启用自动混合精度 |

---

## 8. 数据集要求

### 8.1 Wildtrack数据集结构

```
wildtrack/
├── rectangles.pom                    # 网格参数
├── annotations_positions/            # 3D标签
│   └── *.json                        # 每帧一个JSON
├── Image_subsets/                    # 多视角图像
│   ├── C1/                           # 视角1
│   ├── C2/                           # 视角2
│   ├── ...
│   └── C7/                           # 视角7
└── calibrations/                     # 标定文件
    ├── intrinsic_zero/
    │   └── intr_<CAM>.xml            # 内参
    └── extrinsic/
        └── extr_<CAM>.xml            # 外参
```

### 8.2 注释JSON格式

```json
[
  {
    "positionID": "12345",
    "x": 1.5,
    "y": -2.0,
    ...
  }
]
```

**注意**: `positionID` 会被映射到BEV网格坐标：
- `ix = positionID % NB_WIDTH`
- `iy = positionID // NB_WIDTH`

---

## 9. 常用配置参数

### 9.1 config.py关键参数

```python
# 图像尺寸
DEFAULT_IMG_H = 720           # 输入图像高度
DEFAULT_IMG_W = 1280          # 输入图像宽度
DEFAULT_FEAT_H = 270          # 特征图高度
DEFAULT_FEAT_W = 480          # 特征图宽度

# 网络
DEFAULT_BEV_DOWN = 4          # BEV下采样倍数
DEFAULT_FEAT_CH = 512         # 特征通道数

# 优化器
DEFAULT_LR_INIT = 1e-3        # 初始学习率
DEFAULT_MAX_LR = 0.1          # 最大学习率
DEFAULT_MOMENTUM = 0.5        # SGD动量
DEFAULT_WEIGHT_DECAY = 5e-4   # 权重衰减

# 损失
DEFAULT_ALPHA = 1.0           # 图像损失权重
DEFAULT_MAP_KSIZE = 11        # BEV高斯核大小
DEFAULT_MAP_SIGMA = 2.5       # BEV高斯标准差
DEFAULT_IMG_KSIZE = 11        # 图像高斯核大小
DEFAULT_IMG_SIGMA = 2.0       # 图像高斯标准差
```

### 9.2 输出文件

```
outputs/train_multicam_mvdet_style_v3/
├── model_epoch5.pth           # 第5epoch检查点
├── model_epoch10.pth          # 第10epoch检查点
└── model_final.pth            # 最终模型
```

---

## 附录: 模块文件清单

| 文件 | 路径 | 状态 |
|------|------|------|
| config.py | [src/config.py](file:///workspace/src/config.py) | 活跃 |
| calibration.py | [src/calibration.py](file:///workspace/src/calibration.py) | 活跃 |
| geometry.py | [src/geometry.py](file:///workspace/src/geometry.py) | 活跃 |
| dataset.py | [src/dataset.py](file:///workspace/src/dataset.py) | 活跃 |
| models.py | [src/models.py](file:///workspace/src/models.py) | 活跃 |
| loss.py | [src/loss.py](file:///workspace/src/loss.py) | 活跃 |
| trainer.py | [src/trainer.py](file:///workspace/src/trainer.py) | 活跃 |
| utils.py | [src/utils.py](file:///workspace/src/utils.py) | 活跃 |
| train_main.py | [src/train_main.py](file:///workspace/src/train_main.py) | 活跃 |
| evaluate_main.py | [src/evaluate_main.py](file:///workspace/src/evaluate_main.py) | 活跃 |

---

*文档版本: v1.0*  
*最后更新: 2026-05-07*