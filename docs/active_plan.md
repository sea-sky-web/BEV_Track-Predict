# Active Plan — 当前迭代

> 每轮覆盖此文件。上一轮结果归入 daily-log.md。
> 最后更新：2026-07-08

## 当前状态

| 指标 | 值 | 目标 |
|---|:---:|:---:|
| MODA (best) | 0.857 | ≥ 0.882 |
| MODA (pipeline验证) | 0.849 | — |
| Precision | 0.918 | — |
| Recall | 0.889 | — |
| F1 | 0.903 | — |
| 最优 NMS | 6.0 | — |
| 最优阈值 | 0.400 | — |

**阶段**：第二阶段 — 在现有基线上创新超越 MVDet

## 当前迭代：Focal Loss + Offset Head 消融实验

**目的**：通过隔离实验验证两个改进的独立贡献。

### 代码状态
`feat/focal-loss-offset-head` 分支已完成，代码已通过用户审阅，尚未合并。
消融实验通过 CLI 参数隔离：

| Run | loss_type | offset_weight | use_offset | 隔离的变量 |
|---|---|---|---|---|
| A（回归检查） | mse | 0.0 | false | 无变化，验证代码重构无副作用 |
| B | focal | 0.0 | false | 只有 focal loss |
| C | mse | 1.0 | true | 只有 offset head |
| D（可选） | focal | 1.0 | true | 组合 |

### Pending
- run 28866188056（checkpoint 周期性下载验证，T4）仍在进行中
- A/B/C 三个消融实验需要等 28866188056 完成后依次触发（共用 Colab session 名）

## 当前训练配置（baseline）

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
loss_type: mse (baseline) / focal (实验)
```

## 下一步

1. 等 28866188056 完成 → 验证 checkpoint 下载成功
2. 合并 focal-loss-offset-head 分支
3. 依次触发 A/B/C 消融实验
4. 根据结果决定是否做 D（组合）
