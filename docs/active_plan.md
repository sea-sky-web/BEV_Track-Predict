# Active Plan — 当前迭代

> 最后更新：2026-07-14

## 当前状态 🏆

| 指标 | MVDet baseline | **Ours (best)** | 变化 |
|---|:---:|:---:|:---:|
| MODA | 0.8456 | **0.8918** | **+4.6pp** |
| MODP | 0.7585 | **0.7728** | +1.4pp |
| Precision | 0.9197 | **0.9302** | +1.1pp |
| Recall | 0.8897 | **0.9097** | +2.0pp |
| F1 | 0.9044 | **0.9198** | +1.5pp |
| 参数量 | 32.7M | **5.7M** | **-82.6%** |
| FPS (T4) | 0.62 | **0.96** | **+54.8%** |

**方法**：MobileNet-V2 (truncated, 0.6M) + Learned Attention Fusion (confidence_v2)
**目标达成**：MODA 0.8918 > MVDet 论文 0.882 ✅

## 阶段：第二阶段 — 已完成核心创新，进入论文准备

### 最佳配置
```
backbone: mobilenet_v2 (truncated features[0:14], gradient checkpointing)
fusion_mode: confidence_v2
optimizer: SGD lr=0.1 momentum=0.5 wd=5e-4
scheduler: OneCycleLR max_lr=0.1
epochs: 10, batch=1, frames=360 (train) / 40 (test, frame_start=360)
loss_type: mse
Best threshold: 0.425, Best NMS radius: 6.0
```

## 下一步

### P1（论文支撑实验）
1. MobileNet-V2 + concat (加 gradient ckpt) — 控制变量
2. MobileNet-V2 + geo_confidence_v1 — 验证几何先验在轻量 backbone 下的效果
3. 截断版 MobileNet-V2 推理 benchmark — 更新 FPS 数据
4. 多次训练方差统计（3 runs） — 确认结果置信区间

### P2（可选增强实验）
5. Focal loss 消融 (MobileNet-V2 + cv2)
6. Offset head 消融 (MobileNet-V2 + cv2)
7. 更多 epoch (20/30) — 验证是否欠拟合
