# Active Plan — 当前迭代

> 每轮覆盖此文件。上一轮结果归入 daily-log.md。
> 最后更新：2026-07-10

## 当前状态

| 指标 | 值 | 目标 |
|---|:---:|:---:|
| MODA (best) | 0.857 | ≥ 0.882 |
| MODA (回归检查, Run A) | 0.854 | — |
| Precision | 0.927 | — |
| Recall | 0.895 | — |
| F1 | 0.911 | — |
| 最优 NMS | 6.0-7.0（训练随机性范围） | — |
| 最优阈值 | 0.325-0.40（训练随机性范围） | — |

**阶段**：第二阶段 — 在现有基线上创新超越 MVDet

## 当前迭代：Focal Loss + Offset Head 消融实验

**目的**：通过隔离实验验证两个改进的独立贡献。

### 进度

| Run | loss_type | offset_weight | lr_init | 状态 | MODA |
|---|---|---|---|---|---|
| A（回归检查） | mse | 0.0 | 0.1 | ✅ 完成 | 0.854（正常范围） |
| B（focal only, 首次） | focal | 0.0 | 0.1 | ❌ 爆炸 | N/A |
| BB（focal only, 修复重跑） | focal | 0.0 | 0.1 | ❌ 仍爆炸 | N/A |
| C | mse | 1.0 | 0.1 | ⏸ 未开始 | — |
| D（可选） | focal | 1.0 | ? | ⏸ 未开始 | — |

### Focal Loss 爆炸诊断（两轮根因）

1. **pos_mask bug**（PR #84，已合并）：`tgt.eq(1.0)` 在高斯模糊后几乎不命中 → 修复为 `clamp(max=1.0)` + `ge(1.0-1e-4)`
2. **梯度量级不匹配**（PR #85，待批准合并）：本地实测 focal loss 梯度比 MSE 大 300-3674 倍（`sum()/num_pos` vs `mean()` 归一化差异），SGD lr=0.1 对 focal loss 不安全。CenterNet 原配置 Adam lr=1.25e-4，与我们配置相差 ~800x，量级吻合

### Pending（需用户批准后触发）

- 合并 PR #85 后，用 `--lr_init 0.001` 重跑 focal loss（Run BBB）
- Run BBB 成功后再排 Run C（offset head）
- 消融实验完成后决定是否做 Run D（组合）

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
loss_type: mse (baseline) / focal (实验, 需 lr_init 单独调低)
```

## 下一步

1. 等待用户批准合并 PR #85
2. 用户批准后触发 Run BBB：`loss_type=focal, offset_weight=0.0, lr_init=0.001`
3. Run BBB 成功（loss 正常下降）→ 继续 Run C（offset head）
4. 消融完成后决定是否做 Run D（组合）
5. 待办：FN 空间分布分析，定位 0.854→0.882 差距的 28 个漏检来源（遮挡/覆盖弱区域/训练不足）
