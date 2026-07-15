# Active Plan — 当前迭代

> 最后更新：2026-07-15

## 当前状态 🏆

| 指标 | MVDet baseline | **Ours (best)** | 变化 |
|---|:---:|:---:|:---:|
| MODA | 0.8456 | **0.8950** | **+4.9pp** |
| MODP | 0.7585 | **0.7778** | +1.9pp |
| Precision | 0.9197 | **0.9301** | +1.0pp |
| Recall | 0.8897 | **0.9223** | +3.3pp |
| F1 | 0.9044 | **0.9262** | +2.2pp |
| 参数量 | 32.7M | **5.7M** | **-82.6%** |
| FPS (T4) | 0.62 | **0.96** | **+54.8%** |

**方法**：MobileNet-V2 (truncated, 0.6M) + Geometry-Reliability Attention Fusion (geo_confidence_v1)
**目标达成**：MODA 0.8950 > MVDet 论文 0.882 ✅

## 阶段：第二阶段 — 已完成核心创新，进入论文准备

### 最佳配置
```
backbone: mobilenet_v2 (truncated features[0:14], gradient checkpointing)
fusion_mode: geo_confidence_v1
optimizer: SGD lr=0.1 momentum=0.5 wd=5e-4
scheduler: OneCycleLR max_lr=0.1
epochs: 10, batch=1, frames=360 (train) / 40 (test, frame_start=360)
loss_type: mse
Best threshold: 0.375, Best NMS radius: 5.0
```

## 下一步

### 下一研究模块（计划阶段，尚未实现）

第一模块检测结果冻结后，计划推进 BEV 行人世界坐标 tracking、占用/速度时空场构建与
未来 0.5s / 1.0s / 2.0s 短时预测。详细可行性、理论依据、论文基础、实验矩阵和验收门：

- [Module 2 Plan — BEV 行人时空场映射与短时预测](module2_spatiotemporal_field_prediction_plan.md)

该链接目前仅为研究与实施计划，不代表 tracking 或 forecasting 已进入代码；开始实现前
必须单独评审并更新 `model_definition.md`。

### P1（论文支撑实验）
1. MobileNet-V2 + concat (加 gradient ckpt) — 控制变量
2. MobileNet-V2 + geo_confidence_v1 — 验证几何先验在轻量 backbone 下的效果
3. 截断版 MobileNet-V2 推理 benchmark — 更新 FPS 数据
4. 多次训练方差统计（3 runs） — 确认结果置信区间

### P2（可选增强实验）
5. Focal loss 消融 (MobileNet-V2 + cv2)
6. Offset head 消融 (MobileNet-V2 + cv2)
7. 更多 epoch (20/30) — 验证是否欠拟合
