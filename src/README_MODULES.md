# 模块化架构说明

## 项目结构

成功完成了原始 `08_train_multicam_mvdet_style_v3.py` (1100行) 的模块化重构，分解为 9 个独立模块。

```
scripts/
├── config.py                # 常量和默认参数配置
├── utils.py                 # 工具函数（图像保存、核函数等）
├── loss.py                  # 损失函数（GaussianMSE）
├── calibration.py           # 相机标定处理（读取、缩放、单位推断）
├── geometry.py              # 几何变换（投影矩阵、透视变换）
├── models.py                # 神经网络模型（ResNet、BEVHead、MVDetLikeNet）
├── dataset.py               # 数据加载器（WildtrackMVDetDataset）
├── trainer.py               # 训练器（训练循环、验证、检查点管理）
├── train_main.py            # 主入口（参数解析、初始化、调用训练）
└── verify_modules.py        # 模块验证脚本
```

## 各模块职责

### 1. **config.py** (110 行)
- **职责**: 集中管理所有常量和默认参数
- **内容**:
  - Wildtrack 坐标系常量（ORIGINE_X_M, ORIGINE_Y_M, NB_WIDTH, NB_HEIGHT）
  - 网络超参数（DEFAULT_FEAT_H, DEFAULT_FEAT_W, DEFAULT_BEV_DOWN）
  - 优化器配置（DEFAULT_MAX_LR, DEFAULT_MOMENTUM）
  - 训练策略（DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE）
- **依赖**: 无

### 2. **utils.py** (150 行)
- **职责**: 通用工具函数
- **主要函数**:
  - `save_heat_png()`: 保存热图为 PNG 图像
  - `build_gaussian_kernel_2d()`: 构建高斯卷积核
  - `apply_gaussian_blur()`: 应用高斯模糊
- **依赖**: numpy, torch, PIL

### 3. **loss.py** (170 行)
- **职责**: 定义损失函数
- **主要类**:
  - `GaussianMSE`: MVDet 风格的损失函数（自适应池化 + 高斯模糊 + MSE）
  - `WeightedGaussianMSE`: 带权重的版本（可选）
- **工厂函数**: `create_loss_criterion()`
- **依赖**: torch

### 4. **calibration.py** (280 行)
- **职责**: 处理相机标定参数
- **主要函数**:
  - `parse_rectangles_pom()`: 解析 Wildtrack 配置文件
  - `read_intrinsics()`: 读取相机内参（从 OpenCV XML）
  - `read_opencv_xml_vec()`: 读取向量（rvec, tvec）
  - `scale_intrinsics()`: 根据图像缩放调整内参
  - `decide_unit_scale()`: 自动推断坐标系单位（米/厘米）
- **主要类**: `CalibrationLoader`: 标定数据加载器
- **依赖**: cv2, numpy, config

### 5. **geometry.py** (300 行)
- **职责**: 几何变换和投影
- **主要函数**:
  - `make_worldgrid2worldcoord_mat()`: 网格坐标 → 世界坐标变换矩阵
  - `build_mvdet_proj_mat()`: 构建 MVDet 风格投影矩阵
  - `compute_valid_ratio_from_homography()`: 估计投影有效比例
  - `warp_perspective_torch()`: PyTorch 透视变换
  - `create_grid_sampler()`: 预编译网格采样器
- **依赖**: numpy, torch, calibration

### 6. **models.py** (380 行)
- **职责**: 神经网络模型架构
- **主要类**:
  - `ResNet50Stride8Trunk`: ResNet50 主干（stride=8，使用 dilated conv）
  - `ImgHeadFoot`: 单视角预测头（head/foot 热图）
  - `BEVHeadDilated`: BEV 融合预测头（使用递增 dilation）
  - `MVDetLikeNet`: 完整的多视角 BEV 检测网络
- **工厂函数**: `create_model()`
- **依赖**: torch, torchvision, geometry

### 7. **dataset.py** (280 行)
- **职责**: 数据加载和处理
- **主要类**: `WildtrackMVDetDataset`
  - 加载多视角 RGB 图像
  - 从 3D 标注投影到各视角和 BEV
  - 生成热图标签（head/foot）
- **工厂函数**: `create_wildtrack_dataset()`
- **依赖**: json, numpy, torch, PIL, config

### 8. **trainer.py** (280 行)
- **职责**: 训练流程管理
- **主要类**: `MVDetTrainer`
  - 训练循环（train_epoch）
  - 验证步骤（validate）
  - 检查点管理（save_checkpoint）
  - 可视化（save_visualizations）
- **工厂函数**:
  - `create_optimizer()`: 创建 SGD 优化器
  - `create_scheduler()`: 创建 OneCycle 调度器
- **依赖**: torch, numpy, loss, utils

### 9. **train_main.py** (250 行)
- **职责**: 主入口和组件编排
- **主要函数**:
  - `parse_args()`: 命令行参数解析
  - `main()`: 完整的训练流程
- **流程**:
  1. 解析参数
  2. 加载标定数据
  3. 构建投影矩阵
  4. 创建数据集和 DataLoader
  5. 初始化模型
  6. 创建优化器和调度器
  7. 创建训练器
  8. 执行训练循环
  9. 保存最终模型
- **依赖**: 所有其他模块

## 依赖关系图

```
config.py (无依赖)
    ↓
utils.py ← loss.py
    ↓        ↓
calibration.py
    ↓
geometry.py
    ↓        ↑
dataset.py  models.py
    ↓        ↓
trainer.py ←┤
    ↓        ↓
train_main.py
```

## 低耦合设计特性

### ✅ 单一职责
- 每个模块只负责一个功能域
- 不存在职责重叠

### ✅ 依赖方向
- 所有依赖向下流动（从具体实现到基础工具）
- 没有循环依赖

### ✅ 接口隔离
- 清晰的 import 语句
- 仅导入必要的类和函数

### ✅ 易于扩展
- 新的模型可直接继承 `MVDetLikeNet`
- 新的数据集只需实现 `Dataset` 接口
- 新的损失函数可作为 `GaussianMSE` 的替代品

### ✅ 易于测试
- 每个模块可独立单元测试
- 工厂函数便于依赖注入

## 使用示例

### 验证所有模块
```bash
python scripts/verify_modules.py
```

### 启动训练
```bash
python scripts/train_main.py \
    --data_root wildtrack \
    --views 0,1,2 \
    --epochs 10 \
    --batch 1 \
    --pretrained \
    --device cuda
```

### 仅加载模型进行推理
```python
import sys
sys.path.insert(0, 'scripts')

import torch
from models import create_model
from calibration import CalibrationLoader
from geometry import build_mvdet_proj_mat, make_worldgrid2worldcoord_mat

# 加载模型
proj_mats = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)
model = create_model(3, proj_mats, (135, 240), (270, 480), torch.device("cuda"))

# 推理
x = torch.randn(1, 3, 3, 720, 1280).to("cuda")
map_logits, imgs_logits = model(x)
```

## 关键改进

| 方面 | 原始 | 改进后 |
|------|------|--------|
| 代码行数 | 1100 | 2100 (包含文档) |
| 模块数 | 1 | 9 |
| 可维护性 | 低 | 高 |
| 可测试性 | 低 | 高 |
| 代码重用 | 低 | 高 |
| 修改影响范围 | 全局 | 局部 |

## 验证状态

✅ 所有 9 个模块都已通过验证：
- config.py ✅
- utils.py ✅
- loss.py ✅
- calibration.py ✅
- geometry.py ✅
- models.py ✅
- dataset.py ✅
- trainer.py ✅
- train_main.py ✅

可以直接运行训练脚本。
