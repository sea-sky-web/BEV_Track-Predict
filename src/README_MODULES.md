# src 模块说明

## 当前主链路

`src/` 是唯一活动训练链路，入口如下：
- 训练：`src/train_main.py`
- 评估：`src/evaluate_main.py`

## 模块结构

```
src/
├── config.py
├── utils.py
├── loss.py
├── calibration.py
├── geometry.py
├── models.py
├── dataset.py
├── trainer.py
├── train_main.py
└── evaluate_main.py
```

## 职责概览

- `config.py`：常量与默认超参数（数据、模型、优化器、训练策略）。
- `calibration.py`：标定读取、单位推断、内参缩放、标定缓存。
- `geometry.py`：投影矩阵构建、有效性评估、透视变换实现。
- `dataset.py`：Wildtrack 多视角样本读取与 GT（BEV + per-view）构建。
- `models.py`：MVDetLikeNet（共享 backbone + per-view 辅助头 + BEV 融合头）。
- `loss.py`：`GaussianMSE`（主损失）及可选扩展版本。
- `trainer.py`：训练循环、指标统计、checkpoint 管理。
- `train_main.py`：训练配置编排与流程执行。
- `evaluate_main.py`：离线评估，复用与训练一致的数据/投影链路；支持检测级评估（阈值扫描、Precision/Recall/F1、定位误差）与 `frame_start` 帧切片。
- `utils.py`：热图保存与高斯核构建等通用函数。

## 依赖方向

```
config
  ├─> calibration -> geometry
  ├─> dataset
  ├─> train_main
  └─> evaluate_main

utils -> loss
geometry -> models
loss + models -> trainer
dataset + trainer + geometry + calibration + models -> train_main/evaluate_main
```

## 历史脚本说明

历史探索脚本已迁移至 `archive/legacy/`，不再作为默认入口：
- `archive/legacy/training_prototypes/`
- `archive/legacy/colab_automation_snapshot/`

请结合 `docs/EXPLORATION_MEMORY.md` 与 `archive/legacy/README.md` 查看历史结论与追溯索引。
