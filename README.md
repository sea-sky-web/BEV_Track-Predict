# BEV_Track-Predict

WildTrack 多视角 BEV 行人检测原型。当前阶段只做 BEV heatmap 预测、BEV 点检测和检测指标评估；tracking、ReID、轨迹预测、占用流预测和大型 BEV 框架替换都不在当前范围内。

本仓库的目标是搭建一个可训练、可评估、可复现实验记录的 MVDet-style 闭环：

```text
WildTrack synchronized multi-view images
→ shared pretrained image backbone
→ geometry-based BEV projection
→ multi-view BEV fusion
→ BEV pedestrian heatmap
→ BEV point detections
→ Precision / Recall / F1 / MODA / MODP
```

## 当前默认链路

`src/config.py` 中的当前默认值：

- 数据集：WildTrack
- 视角：全部 7 个视角 `0,1,2,3,4,5,6`
- 帧数：`max_frames=-1`，使用全部帧
- backbone：`resnet18`
- pretrained：`true`，使用 ImageNet 预训练权重
- fusion：`concat`（MVDet 原始方式，7 视角特征拼接）
- BEV head：`MVDetMapClassifier`（3 层 dilated conv，无 BN，bias=False）
- batch：`1`
- optimizer：SGD（lr=0.05, momentum=0.5, weight_decay=5e-4）
- scheduler：OneCycleLR（max_lr=0.05）
- epochs：10
- `freeze_backbone_epochs=0`
- augmentation：默认启用颜色抖动；水平翻转默认 `0.0`
- evaluation：阈值扫描 + Precision / Recall / F1 / MODA / MODP / localization error
- NMS：贪心距离抑制，半径 20 cells（2.0m）

`confidence_v2`、`confidence_v1`、Adam、CosineAnnealingLR 和 ResNet-50 仍保留为显式 legacy/ablation 选项。

## 安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 给出最小依赖约束。`torch` / `torchvision` 建议按本机 CUDA 或 Colab 环境单独选择匹配版本安装。

## WildTrack 数据目录

默认数据根目录是 `wildtrack`。期望结构：

```text
wildtrack/
├── rectangles.pom
├── annotations_positions/
│   └── *.json
├── Image_subsets/
│   ├── C1/
│   ├── C2/
│   └── ...
└── calibrations/
    ├── intrinsic_zero/
    │   └── intr_<CAM>.xml
    └── extrinsic/
        └── extr_<CAM>.xml
```

每个 annotation JSON 至少需要包含 `positionID`。图像文件 stem 需要和 annotation stem 一致，支持 `.png`、`.jpg`、`.jpeg`。

## 训练

推荐从仓库根目录运行：

```bash
python src/train_main.py \
  --data_root wildtrack \
  --device cuda
```

等价的显式当前默认命令：

```bash
python src/train_main.py \
  --data_root wildtrack \
  --views 0,1,2,3,4,5,6 \
  --max_frames -1 \
  --batch 1 \
  --backbone resnet18 \
  --pretrained true \
  --fusion_mode concat \
  --augment true \
  --augment_hflip_prob 0.0 \
  --augment_color_jitter 0.2,0.2,0.2,0.05 \
  --optimizer sgd \
  --scheduler onecycle \
  --lr_init 0.05 \
  --weight_decay 0.0005 \
  --freeze_backbone_epochs 0 \
  --device cuda
```

输出：

- 默认目录：`outputs/train_multicam_mvdet_style_v3`
- 每 5 个 epoch 保存一次 checkpoint
- 最终模型：`outputs/train_multicam_mvdet_style_v3/model_final.pth`

## 评估

Loss-level 评估：

```bash
python src/evaluate_main.py \
  --data_root wildtrack \
  --views 0,1,2,3,4,5,6 \
  --backbone resnet18 \
  --fusion_mode concat \
  --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth \
  --device cuda
```

检测级评估：

```bash
python src/evaluate_main.py \
  --data_root wildtrack \
  --views 0,1,2,3,4,5,6 \
  --backbone resnet18 \
  --fusion_mode concat \
  --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth \
  --device cuda \
  --report_detection \
  --metrics_out outputs/eval_metrics.json
```

检测输出包含：

- `det_precision`
- `det_recall`
- `det_f1`
- `det_moda`
- `det_modp`
- `det_loc_err_m`
- `det_tp` / `det_fp` / `det_fn`
- extraction config，包括阈值、NMS、最大预测数和 MODA matching distance

训练 loss 不能单独作为模型改进证据。是否改进必须依赖相同配置下的正式评估指标对比。

## Colab 实验闭环

Colab 默认配置：

```text
configs/exp_colab.yaml
```

运行：

```bash
python scripts/run_colab_exp.py
```

正式实验记录应写入：

```text
ai_runs/YYYYMMDD_HHMMSS/
├── ai_context.md
├── metrics.json
├── train_tail.log
└── error.log
```

每个 `ai_context.md` 必须记录 backbone、pretrained、fusion、augmentation、optimizer、scheduler、views、max_frames、loss 配置和检测指标。不要在正式 Colab 结果确认前宣称 F1/MODA 达标。

## 可视化工具

投影覆盖检查：

```bash
python scripts/visualize_projection.py \
  --data_root wildtrack \
  --views 0,1,2,3,4,5,6
```

`confidence_v2` per-view 权重图：

```bash
python scripts/visualize_fusion_weights.py \
  --data_root wildtrack \
  --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth
```

这些脚本需要真实 WildTrack 数据或真实 checkpoint。不要用手绘或合成图作为几何/融合验证证据。

## 无数据验证

静态检查：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/bevtrack_pycache python -m compileall src scripts tests
```

测试：

```bash
PYTHONPATH=src pytest \
  tests/test_geometry.py \
  tests/test_metrics.py \
  tests/test_augmentation.py \
  tests/test_smoke_forward.py \
  -v
```

GitHub Actions 的 `Python Smoke` workflow 会安装 CPU 版 PyTorch，并覆盖几何、MODA/MODP、augmentation 和 ResNet-18 forward/backward smoke。

## 代码结构

- `src/config.py`：默认参数和 WildTrack 常量
- `src/augmentation.py`：训练增强
- `src/calibration.py`：标定读取和单位推断
- `src/geometry.py`：投影矩阵、valid ratio、torch warp
- `src/dataset.py`：WildTrack 多视角数据集和 GT 构建
- `src/models.py`：ResNet backbone、BEV projection、fusion、heads
- `src/loss.py`：Gaussian MSE loss
- `src/metrics.py`：MODA/MODP 和检测统计
- `src/trainer.py`：训练循环、优化器、scheduler、checkpoint
- `src/train_main.py`：训练入口
- `src/evaluate_main.py`：评估入口
- `scripts/run_colab_exp.py`：Colab launcher
- `scripts/commit_ai_runs.py`：实验记录归档
- `scripts/visualize_projection.py`：投影可视化
- `scripts/visualize_fusion_weights.py`：融合权重可视化

## 文档

- 训练目标与预期效果：`docs/training_goals.md`
- MVDet 对齐修复计划：`docs/mvdet_alignment_plan.md`
- 模型边界：`docs/model_definition.md`
- 实验协议：`docs/experiment_protocol.md`
- 评估管线分析：`docs/eval_pipeline_analysis.md`
- 迭代记录格式：`docs/experiment_iteration_protocol.md`
- 数据契约：`docs/dataset_contract.md`
- 历史探索：`docs/EXPLORATION_MEMORY.md`

## 常见问题

- 找不到模块：从仓库根目录运行命令，或设置 `PYTHONPATH=src`
- 找不到 `annotations_positions`：检查 WildTrack 数据根目录
- 找不到标定 XML：检查 `wildtrack/calibrations/intrinsic_zero` 和 `wildtrack/calibrations/extrinsic`
- checkpoint shape mismatch：训练和评估的 `--views`、`--backbone`、`--fusion_mode`、`--bev_down`、`--feat_h`、`--feat_w` 必须一致
- 本地 pytest 缺依赖：先安装 `requirements.txt`，或等待 GitHub Actions smoke
