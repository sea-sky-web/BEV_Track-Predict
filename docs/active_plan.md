# Active Plan — 当前迭代

> 每轮覆盖此文件。上一轮结果归入 daily-log.md。
> 最后更新：2026-07-07

## 当前状态

| 指标 | 值 | 目标 |
|---|:---:|:---:|
| MODA | 0.857 | ≥ 0.882 |
| Precision | 0.918 | — |
| Recall | 0.889 | — |
| F1 | 0.903 | — |
| 最优 NMS | 6.0 | — |
| 最优阈值 | 0.400 | — |

## 当前训练配置

```
backbone: resnet18 (progressive dilation: L3.B1=2, L4.B0=2, L4.B1=4)
fusion_mode: concat
img_head mid_ch: 64
optimizer: SGD lr=0.1 momentum=0.5 wd=5e-4
scheduler: OneCycleLR max_lr=0.1
epochs: 10, batch=1, frames=360 (train) / 40 (test, frame_start=360)
augment: false
amp: false
bev_pos_weight: 1.0
NMS: det_min_distance=6.0 (reduced grid cells, 0.6m physical)
MODA matching: 0.5m (Hungarian)
```

## Pending 任务

**Pipeline 验证 run 28845973141**（L4, 进行中）
- 架构与 MVDet 完全对齐（PR #77: progressive dilation + img_head mid_ch=64）
- 预期：MODA 接近 0.882 则 pipeline 验证通过

## 当前迭代：Pipeline 验证

**目的**：证明 pipeline 完全正确。将架构与 MVDet 完全一致后，MODA 应接近 0.882。

**判断标准**：
- MODA ≥ 0.87 → pipeline 验证通过，进入创新阶段（路线 B）
- MODA 0.85-0.87 → 差距可接受，可能是训练随机性或 epoch 不足
- MODA < 0.85 → 仍有未知差异，需继续排查

## MVDet 对齐完整差异清单（已全部修复）

| 差异 | 修复 PR | 影响 |
|---|---|---|
| BEV H/W 转置 (NB_WIDTH/HEIGHT) | 06-29 | MODA 0→0.5 |
| GT 坐标映射 (positionID) | 06-29 | 同上 |
| Gaussian sigma (5.0→√5) | 06-29 | 同上 |
| lr 0.05→0.1 | #66 | 配置对齐 |
| grad_clip 移除 | #66 | 配置对齐 |
| bev_pos_weight 10→1 | #69 | 配置对齐 |
| eval frame_start 1800→360 | #68 | 数据泄露修复 |
| workflow 默认值 (max_frames, pos_weight) | #72 | 默认值对齐 |
| AMP 移除 | #72 | 配置对齐 |
| augmentation 禁用 | #72 | 配置对齐 |
| **NMS 半径 4× 过大** (20→5 reduced cells) | #73 | **MODA 0.44→0.79** |
| NMS 半径微调 (5→6) | 网格扫描 | MODA 0.79→0.857 |
| backbone dilation 渐进模式 | #77 | 架构对齐 |
| img_head mid_ch 128→64 | #77 | 架构对齐 |

## 下一步（待验证结果后决定）

- 如果通过 → 切换到路线 B：在现有框架上创新超越 MVDet
  - 候选方向：confidence fusion、增加 epoch、attention 机制
- 如果未通过 → 深入对比 backbone 输出特征、loss 数值、训练曲线
