# Plan: Focal Loss + Offset Head

## Context

Phase 2 第一步：从 MVDet baseline (MODA 0.849) 出发，替换 Gaussian MSE 为 CenterNet-style Penalty-Reduced Focal Loss，同时添加 Offset 回归头。这是当前所有 SOTA 方法（QMVDet, EarlyBird, PVH, MSMVD）的标配，预期 +2-5% MODA。

## 关键变化

### 1. `src/loss.py` — 新增 `PenaltyReducedFocalLoss`

```python
class PenaltyReducedFocalLoss(nn.Module):
    """CenterNet-style focal loss for heatmap regression."""
    def __init__(self, alpha=2.0, beta=4.0):
        # alpha: focal power on hard examples
        # beta: penalty reduction power near GT centers

    def forward(self, pred_logits, target, kernel):
        # 1. Pool + Gaussian smooth target (复用现有逻辑)
        # 2. pred = sigmoid(pred_logits)
        # 3. 正样本 (target == 1): -(1-pred)^alpha * log(pred)
        # 4. 负样本 (target < 1): -(1-target)^beta * pred^alpha * log(1-pred)
        # 5. 返回 mean over all pixels, normalized by num_positives
```

接口与 `GaussianMSE` 兼容（同样接收 pred, target, kernel），drop-in 替换。

### 2. `src/models.py` — 新增 Offset Head

在 `MVDetLikeNet` 中添加 `offset_head`：
- 与 `bev_head` 并行，共享 `bev_fused` 输入
- 输出 (B, 2, Hb, Wb) — 每个像素预测 (dx, dy) 偏移
- 结构：Conv2d(in_ch, 64, 3, pad=1) → ReLU → Conv2d(64, 2, 1)

`forward()` 返回值变为 `(map_logits, offset_preds, imgs_logits)`。

### 3. `src/trainer.py` — 整合新 loss

- BEV loss 切换到 `PenaltyReducedFocalLoss`
- 新增 offset L1 loss（仅在 GT 正样本位置计算）
- 总 loss = focal_loss + lambda_offset * offset_l1 + alpha * img_loss
- `lambda_offset` 默认 1.0

### 4. `src/evaluate_main.py` — 使用 offset 修正检测点

在 `_extract_points` 后，用 offset 预测修正 (y, x) 坐标：
```python
y_refined = y + offset[0, y, x]
x_refined = x + offset[1, y, x]
```

### 5. `src/config.py` — 新增参数

```python
DEFAULT_LOSS_TYPE = "focal"  # "mse" | "focal"
DEFAULT_FOCAL_ALPHA = 2.0
DEFAULT_FOCAL_BETA = 4.0
DEFAULT_OFFSET_WEIGHT = 1.0
```

### 6. `scripts/colab_train.py` — 传参

添加 `--loss_type focal` 到训练命令。保留 `--loss_type mse` 用于对比实验。

## 不改动

- 投影矩阵、backbone、fusion 模式不变
- img_loss 保持 GaussianMSE（辅助监督不受影响）
- 评估协议（NMS 半径、阈值扫描）不变

## 向后兼容

- `--loss_type mse` 恢复原 GaussianMSE 行为
- 旧 checkpoint（无 offset_head）加载时 `strict=False` + 警告

## 验证

1. 本地 smoke test: `python src/train_main.py --max_frames 2 --epochs 1 --loss_type focal --device cpu`
2. 触发 Colab train+eval run（需用户批准）
3. 对比 focal vs mse 的 MODA
