# BEV_Track&Predict

Current stage: only BEV pedestrian detection and point extraction are active.
Tracking and prediction are later-stage tasks and are not part of the current model definition.

本仓库目标：在 Wildtrack 数据集上快速搭建并跑通一个“多视角→BEV→行人占据/定位(POM)→后续关联与预测”的可训练闭环原型。优先工程闭环与可复现实验，不走长理论铺垫路线。

## 快速开始

### 环境依赖

```bash
pip install -r requirements.txt
```

其中 `torch/torchvision` 需要你按本机 CUDA/驱动情况安装匹配版本（`requirements.txt` 里给的是最小约束：`torch>=2.0`、`torchvision>=0.15`）。

### 数据集目录（Wildtrack）

`src/train_main.py` 默认使用的数据根目录是 `wildtrack`（见 `src/config.py` 的 `DEFAULT_DATA_ROOT`）。代码期望如下结构：

- `wildtrack/rectangles.pom`
- `wildtrack/annotations_positions/*.json`
  - 每个样本的 JSON 至少需要包含 `positionID` 字段；该字段会被映射到 BEV 网格 `(ix, iy)`，用于生成 BEV 主监督热图
- `wildtrack/Image_subsets/C1..C7/`
  - 每个选定视角会读取对应目录里的图像：文件名 stem 需要与 `annotations_positions/*.json` 的 stem 一致
  - 支持扩展名：`.png/.jpg/.jpeg`
- `wildtrack/calibrations/`
  - `intrinsic_zero/intr_<CAM>.xml`
  - `extrinsic/extr_<CAM>.xml`
  - 其中 `<CAM>` 来自 `src/config.py` 的 `CAM_NAMES`

### 训练（当前主入口）

训练入口脚本：`src/train_main.py`

从仓库根目录执行（推荐）：

```bash
python src/train_main.py --data_root wildtrack --device cuda
```

常用参数：

- `--views`：多视角 ID，默认使用全部 7 个 WildTrack 视角 `0,1,2,3,4,5,6`
- `--epochs`：默认 `10`
- `--max_frames`：默认 `-1`，表示使用全部帧；可设为小正数进行 smoke test
- `--batch`：默认 `1`
- `--bev_down`：默认 `4`
- `--pretrained true|false` / `--no-pretrained`：默认使用 ImageNet 预训练 ResNet-50 backbone
- `--optimizer`：默认 `adam`；`sgd` 保留给 legacy 复现实验
- `--scheduler`：默认 `cosine`；`onecycle` 保留给 legacy 复现实验
- `--lr_init` / `--weight_decay`：Adam 默认分别为 `1e-4` / `1e-4`
- `--freeze_backbone_epochs`：默认 `3`，前 3 个 epoch 冻结共享 image backbone
- `--amp`：启用自动混合精度（仅 `cuda` 时有效）
- `--drop_bad_views`：丢弃投影 `valid_ratio` 低于阈值的视角
- `--valid_thr`：默认 `0.05`
- `--momentum` / `--max_lr`：SGD + OneCycle legacy 复现实验参数

输出目录与模型文件：

- 默认输出目录：`outputs/train_multicam_mvdet_style_v3`（见 `src/config.py` 的 `DEFAULT_OUTPUT_DIR`）
- 每 5 个 epoch 保存一次：`model_epoch{epoch}.pth`
- 训练结束保存最终模型：`model_final.pth`（`state_dict`）

训练过程中会打印：

- 每个视角的投影 `valid_ratio`
- step 级别的 `loss/bev/img/pos_mse/aux_pos_mse`（频率由 `--log_every` 控制）

### 评估入口

评估脚本：`src/evaluate_main.py`

```bash
python src/evaluate_main.py --data_root wildtrack --views 0,1,2,3,4,5,6 --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth --device cuda
```

如需做“检测级”评估（阈值扫描 + Precision/Recall/F1 + 定位误差），可直接启用：

```bash
python src/evaluate_main.py --data_root wildtrack --views 1,2 --drop_bad_views --valid_thr 0.15 --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth --device cuda --frame_start 300 --max_frames 100 --report_detection --metrics_out outputs/eval_metrics.json
```

该脚本与 `src/train_main.py` 使用相同的投影与数据构建链路，输出 `loss/bev_loss/img_loss/pos_mse/aux_pos_mse` 以及模型参数统计。

## 历史探索归档

- 历史训练原型：`archive/legacy/training_prototypes/`
- Colab 自动化快照：`archive/legacy/colab_automation_snapshot/`
- 归档索引：`archive/legacy/README.md`
- 探索记忆文档：`docs/EXPLORATION_MEMORY.md`

`scripts/` 目录不再承载历史训练入口，仅保留活动工具脚本（见 `scripts/README.md`）。

## 代码模块对应关系

- 数据加载与标签生成：`src/dataset.py`
- 标定与投影相关：`src/calibration.py`
- 网络结构：`src/models.py`
- 训练循环与检查点：`src/trainer.py`
- 主训练脚本/参数入口：`src/train_main.py`

## 常见问题

- 找不到模块/导入失败：请从仓库根目录用 `python src/train_main.py ...` 运行
- 报错提示 `annotations_positions` 不存在：请检查数据集目录结构
- 报错提示标定 XML 不存在：请检查 `wildtrack/calibrations/intrinsic_zero` 与 `wildtrack/calibrations/extrinsic` 下的文件命名
