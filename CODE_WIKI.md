# BEV_Track&Predict 项目 Code Wiki

## 项目概述

BEV_Track&Predict 是一个基于 WildTrack 数据集的多视角 Bird's Eye View (BEV) 行人检测深度学习研究项目。该项目实现了几何引导的 MVDet 风格多视角 BEV 行人检测器，支持多摄像头视角融合、BEV 热力图预测和行人点提取。

### 核心功能

- 多视角图像到 BEV 空间的透视投影
- 空间感知的多视角置信度融合
- BEV 行人占据热力图预测
- BEV 行人定位点提取
- 检测级评估（Precision/Recall/F1）

---

## 项目架构

### 目录结构

```
BEV_Track&Predict/
├── src/                    # 核心源代码目录
│   ├── config.py           # 配置与常量定义
│   ├── dataset.py          # 数据加载与标签生成
│   ├── calibration.py       # 相机标定处理
│   ├── geometry.py          # 几何投影与透视变换
│   ├── models.py           # 神经网络模型定义
│   ├── loss.py             # 损失函数定义
│   ├── trainer.py          # 训练循环与检查点管理
│   ├── utils.py            # 通用工具函数
│   ├── train_main.py       # 训练入口脚本
│   └── evaluate_main.py    # 评估入口脚本
├── configs/                # 配置文件目录
├── scripts/                # 辅助脚本
├── docs/                   # 文档目录
├── ai_runs/                # 实验记录目录
├── archive/                # 历史归档
├── requirements.txt        # Python依赖
├── README.md               # 项目说明
└── AGENTS.md               # AI助手规范
```

---

## 核心模块详解

### 1. config.py - 配置管理

**职责**: 定义项目常量、默认超参数和数据路径配置。

**关键配置项**:

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DEFAULT_DATA_ROOT` | 数据集根目录 | `wildtrack` |
| `DEFAULT_OUTPUT_DIR` | 输出目录 | `outputs/train_multicam_mvdet_style_v3` |
| `CAM_NAMES` | 摄像头名称列表 | `['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']` |
| `BEV_SHAPE` | BEV 空间尺寸 | `(480, 480)` |
| `WORLD_SIZE` | 实际世界尺寸(米) | `(16.8, 9.6)` |

**关键常量**:

```python
CAM_NAMES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
BEV_SHAPE = (480, 480)      # BEV 热力图尺寸
WORLD_SIZE = (16.8, 9.6)    # 实际世界尺寸 (W, H) 单位：米
GROUND_HEIGHT = 0.0         # 地面高度假设
```

### 2. calibration.py - 相机标定

**职责**: 读取并处理相机内外参，支持标定文件缓存和内参缩放。

**主要函数**:

| 函数 | 功能 |
|------|------|
| `load_intrinsics(camera_name, base_path)` | 加载内参矩阵 |
| `load_extrinsics(camera_name, base_path)` | 加载外参矩阵 |
| `get_camera_matrix(intrinsic_file)` | 解析内参 XML 文件 |
| `get_extrinsic_matrix(extrinsic_file)` | 解析外参 XML 文件 |
| `scale_intrinsics(K, scale_factor)` | 按图像缩放因子调整内参 |

**数据类型**:

- **内参矩阵**: 3x3 矩阵，包含焦距 (fx, fy) 和主点 (cx, cy)
- **外参矩阵**: 3x4 矩阵，包含旋转矩阵 R 和平移向量 T
- **格式**: OpenCV XML 格式

### 3. geometry.py - 几何投影

**职责**: 实现多视角图像到 BEV 空间的几何投影核心算法。

**主要函数**:

| 函数 | 功能 |
|------|------|
| `build_projection_matrixes(...)` | 构建多视角投影矩阵集合 |
| `compute_bev_transform(cam_params, bev_shape, world_size)` | 计算单视角 BEV 变换矩阵 |
| `warp_image_to_bev(image, M, output_shape)` | 透视变换图像到 BEV 空间 |
| `validate_projection(proj_points, bev_shape)` | 验证投影有效性 |
| `project_points_to_bev(points_3d, projections)` | 3D 点投影到 BEV 平面 |

**投影流程**:

```
多视角相机 → 外参(R|T) → 相机坐标系
                        ↓
              内参矩阵(K) → 像素坐标系
                        ↓
              透视变换 → BEV 网格坐标系
```

**关键参数**:

```python
class CameraParameters:
    K: np.ndarray  # 3x3 内参矩阵
    R: np.ndarray  # 3x3 旋转矩阵
    T: np.ndarray  # 3x1 平移向量
    M: np.ndarray  # 3x4 投影矩阵 = K @ [R|T]
```

### 4. dataset.py - 数据加载

**职责**: 加载 WildTrack 数据集，构建训练样本和标签。

**主要类**:

| 类 | 功能 |
|----|------|
| `WildtrackDataset` | WildTrack 数据集加载器 |

**关键方法**:

| 方法 | 功能 |
|------|------|
| `__init__(root, views, transform=None)` | 初始化数据集 |
| `__len__()` | 返回样本数量 |
| `__getitem__(idx)` | 获取单个样本 |
| `load_annotations(frame_id)` | 加载帧标注 |
| `load_image(frame_id, view_id)` | 加载指定视角图像 |
| `build_bev_label(positions)` | 构建 BEV 监督热力图 |

**样本结构**:

```python
{
    'images': List[np.ndarray],           # 多视角图像列表
    'bev_gt': np.ndarray,                 # BEV 真实热力图 (H, W)
    'view_gts': List[np.ndarray],         # 各视角辅助热力图
    'valid_ratios': List[float],         # 各视角投影有效率
    'frame_id': int                        # 帧 ID
}
```

### 5. models.py - 神经网络模型

**职责**: 定义 MVDet 风格的多视角 BEV 检测网络。

**主要类**:

| 类 | 功能 |
|----|------|
| `ImageEncoder` | 共享图像编码器 (ResNet18 backbone) |
| `BevDecoder` | BEV 解码器 (多层卷积) |
| `ViewHead` | 单视角辅助头 |
| `BevHead` | BEV 主预测头 |
| `MVDetLikeNet` | 完整的多视角 BEV 检测网络 |

**MVDetLikeNet 网络结构**:

```
输入: List[Images] (多视角)
         ↓
    ImageEncoder (共享 ResNet18)
         ↓
    ┌────┼──────────────────────┐
    ↓    ↓    ↓                  ↓
ViewHead ViewHead ... ViewHead   BevDecoder
    ↓    ↓    ↓                  ↓
view_gt view_gt ... view_gt    BevHead
                                    ↓
                               bev_heatmap
```

**输出**:

| 输出张量 | 形状 | 说明 |
|----------|------|------|
| `bev_heatmap` | (B, 1, H, W) | BEV 占据热力图 |
| `view_heatmaps` | List[(B, 1, H, W)] | 各视角辅助热力图 |

### 6. loss.py - 损失函数

**职责**: 定义训练损失函数。

**主要类/函数**:

| 类/函数 | 功能 |
|---------|------|
| `GaussianMSE` | 高斯 MSE 损失(主损失) |
| `gaussian_2d(shape, center, sigma)` | 生成 2D 高斯核 |

**GaussianMSE 损失**:

```python
class GaussianMSE(nn.Module):
    def __init__(self, target_sigma=2.0):
        super().__init__()
        self.sigma = target_sigma
    
    def forward(self, pred, target):
        # pred: (B, 1, H, W) 预测热力图
        # target: (B, 1, H, W) 高斯热力图标签
        return F.mse_loss(pred, target)
```

**损失组成**:

| 损失项 | 权重 | 说明 |
|--------|------|------|
| `bev_loss` | 1.0 | BEV 主损失 |
| `img_loss` | 0.1 | 图像空间辅助损失 |
| `pos_mse` | - | 预测点与 GT 点 MSE |

### 7. trainer.py - 训练循环

**职责**: 管理训练过程、指标统计和模型检查点。

**主要类**:

| 类 | 功能 |
|----|------|
| `Trainer` | 训练器封装 |

**关键方法**:

| 方法 | 功能 |
|------|------|
| `train_epoch(model, dataloader)` | 单 epoch 训练 |
| `validate(model, dataloader)` | 验证集评估 |
| `save_checkpoint(epoch)` | 保存检查点 |
| `load_checkpoint(path)` | 加载检查点 |

**检查点管理**:

```python
# 保存格式
model_epoch{epoch}.pth      # 中间检查点(每5个epoch)
model_final.pth             # 最终模型(state_dict)
```

### 8. utils.py - 工具函数

**职责**: 提供通用工具函数。

**关键函数**:

| 函数 | 功能 |
|------|------|
| `generate_heatmap(shape, positions, sigma)` | 生成高斯热力图 |
| `draw_points_on_bev(bev, points, color)` | 在 BEV 上绘制点 |
| `save_heatmap(img, path)` | 保存热力图可视化 |
| `set_seed(seed)` | 设置随机种子 |

### 9. train_main.py - 训练入口

**职责**: 训练流程的入口脚本，协调各模块。

**命令行参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | `wildtrack` | 数据集根目录 |
| `--views` | str | `0,1,2` | 视角 ID 列表 |
| `--epochs` | int | 10 | 训练轮数 |
| `--max_frames` | int | 300 | 最大帧数(-1表示全部) |
| `--batch` | int | 1 | 批大小 |
| `--bev_down` | int | 4 | BEV 下采样因子 |
| `--device` | str | `cuda` | 设备(cuda/cpu) |
| `--amp` | flag | False | 启用混合精度 |
| `--lr_init` | float | 0.01 | 初始学习率 |
| `--valid_thr` | float | 0.10 | 有效投影阈值 |

### 10. evaluate_main.py - 评估入口

**职责**: 离线评估模型的检测性能。

**命令行参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | 必需 | 模型路径 |
| `--report_detection` | flag | False | 输出检测级指标 |
| `--metrics_out` | str | - | 指标输出文件 |
| `--frame_start` | int | 0 | 起始帧 |
| `--threshold` | float | 0.5 | 检测阈值 |

**评估指标**:

| 指标 | 说明 |
|------|------|
| `loss` | 总损失 |
| `bev_loss` | BEV 损失 |
| `img_loss` | 视角损失 |
| `pos_mse` | 定位 MSE |
| `Precision` | 精确率 |
| `Recall` | 召回率 |
| `F1` | F1 分数 |

---

## 依赖关系图

```
                    ┌─────────────┐
                    │  config.py  │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │calibration│  │  geometry   │  │   dataset  │
    └─────┬──────┘  └──────┬─────┘  └─────┬──────┘
          │               │             │
          └───────┬───────┘             │
                  ↓                     ↓
           ┌────────────┐         ┌────────────┐
           │   models   │         │    loss    │
           └──────┬─────┘         └──────┬─────┘
                  │                     │
                  └──────────┬──────────┘
                             ↓
                      ┌────────────┐
                      │  trainer   │
                      └──────┬─────┘
                             │
              ┌──────────────┼──────────────┐
              ↓                             ↓
       ┌────────────┐                 ┌────────────┐
       │train_main  │                │evaluate_main│
       └────────────┘                 └────────────┘
```

---

## 运行方式

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确认 PyTorch CUDA 版本匹配
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 训练模型

```bash
# 基本训练命令
python src/train_main.py --data_root wildtrack --views 0,1,2 --device cuda

# 完整参数示例
python src/train_main.py \
    --data_root wildtrack \
    --views 0,1,2,3,4 \
    --epochs 20 \
    --max_frames 500 \
    --batch 2 \
    --device cuda \
    --amp \
    --lr_init 0.005 \
    --bev_down 4 \
    --valid_thr 0.15
```

### 评估模型

```bash
# 基本评估
python src/evaluate_main.py \
    --data_root wildtrack \
    --views 0,1,2 \
    --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth \
    --device cuda

# 检测级评估(输出 Precision/Recall/F1)
python src/evaluate_main.py \
    --data_root wildtrack \
    --views 1,2 \
    --model_path outputs/model_final.pth \
    --device cuda \
    --report_detection \
    --metrics_out outputs/eval_metrics.json \
    --frame_start 300 \
    --max_frames 100
```

### 快速验证

```bash
# CPU 快速冒烟测试
python src/train_main.py --data_root wildtrack --views 0,1 --device cpu --max_frames 2
```

---

## 数据集结构

### WildTrack 数据集目录

```
wildtrack/
├── rectangles.pom                 # 场景边界定义
├── annotations_positions/         # 行人位置标注
│   ├── 0000.json                   # 帧 0 的标注
│   ├── 0001.json
│   └── ...
├── Image_subsets/
│   ├── C1/                         # 视角 1 图像
│   │   ├── 0000.png
│   │   └── ...
│   ├── C2/
│   └── ...
└── calibrations/
    ├── intrinsic_zero/             # 内参目录
    │   ├── intr_C1.xml
    │   └── ...
    └── extrinsic/                  # 外参目录
        ├── extr_C1.xml
        └── ...
```

### 标注文件格式

```json
{
  "positionID": 1,
  "sceneName": "wildtrack",
  "x": 5.2,
  "y": 3.1,
  "z": 0.0
}
```

---

## 模型定义边界

根据 `docs/model_definition.md`，当前模型边界定义如下：

### 包含功能

- BEV 行人检测
- BEV 热力图预测
- BEV 点提取
- 检测评估
- 实验日志

### 排除功能

- 目标跟踪 (Tracking)
- ReID 特征提取
- 轨迹预测
- 占用流预测
- 群体预测
- 交通决策
- 合成数据训练
- 实车部署

---

## 关键技术点

### 1. 几何投影原理

项目使用针孔相机模型进行多视角到 BEV 的投影：

```
世界坐标系 → 相机坐标系 → 图像坐标系 → BEV 坐标系
   (X,Y,Z)      [R|T]        [K]        透视变换
```

### 2. 多视角融合策略

采用空间感知的置信度融合：

1. 各视角独立编码提取特征
2. 计算各视角在 BEV 网格的有效投影
3. 基于有效率加权融合各视角预测
4. BEV 解码器输出最终热力图

### 3. 热力图监督

使用 2D 高斯分布作为回归目标：

```python
# 行人位置生成高斯热力图
heatmap = Gaussian2D(bev_shape, center, sigma=2.0)
```

---

## 实验记录规范

实验记录存储在 `ai_runs/YYYYMMDD_HHMMSS/` 目录，每个实验包含：

| 文件 | 说明 |
|------|------|
| `ai_context.md` | 实验上下文与迭代记录 |
| `metrics.json` | 关键指标 JSON |
| `train_tail.log` | 训练日志尾部 |
| `error.log` | 错误日志(如有问题) |

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError` | 从仓库根目录运行 `python src/train_main.py` |
| `annotations_positions` 找不到 | 检查数据集目录结构 |
| 标定 XML 不存在 | 确认 `wildtrack/calibrations/` 下的文件命名 |
| CUDA OOM | 减小 `batch_size` 或 `--max_frames` |
| 训练 loss 不下降 | 检查投影矩阵和数据标签构建 |

---

## 参考文档

- `docs/model_definition.md` - 模型定义规范
- `docs/experiment_iteration_protocol.md` - 实验迭代协议
- `docs/EXPLORATION_MEMORY.md` - 探索历史记忆
- `archive/legacy/README.md` - 历史原型索引
