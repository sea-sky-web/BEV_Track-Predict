# 正样本加权 + 验证指标修正

## Context

三步训练坍缩修复后，模型从"完全不学"变为"能学"。但 pos_mse 指标存在设计缺陷（sigmoid/logit 空间不匹配，理论下界 0.073），导致我们误以为模型还有大量提升空间，实际已接近该指标天花板。同时，GaussianMSE 损失未加权，正样本仅占 BEV 地图约 5%，梯度被背景主导。

**目标**：启用正样本加权抬高行人响应 + 用直接反映检测能力的指标替换 pos_mse。

---

## 修改 1：启用正样本加权

### 1.1 `src/config.py` — 添加默认权重常量

```python
DEFAULT_BEV_POS_WEIGHT = 10.0
DEFAULT_BEV_NEG_WEIGHT = 1.0
```

10.0 是基于正负比例（~5% / ~95% ≈ 20x）的保守起点。

### 1.2 `src/train_main.py` — 引用 config 常量

将 `--bev_pos_weight` 和 `--bev_neg_weight` 的 `default=1.0` 改为引用 config 常量。无需改 Trainer 调用链（已有完整传递路径）。

### 1.3 `scripts/colab_train.py` — 传递权重参数

- 添加 `--bev_pos_weight` CLI 参数
- 在 `train_cmd` 列表中加入 `"--bev_pos_weight", str(args.bev_pos_weight)`

### 1.4 `.github/workflows/colab-train.yml` — 工作流输入

添加 `bev_pos_weight` 输入项（default: '10.0'），传入 colab_train.py。

---

## 修改 2：替换验证指标

### 核心思路

当前 pos_mse 的问题：`((sigmoid(logits) - binary_GT)²)[pos]`
- sigmoid 和 binary 不在同一空间，有 0.073 不可消除的 floor

新指标全部基于 **raw logits vs Gaussian-smoothed GT**（与 loss 同空间）：

| 指标 | 计算 | 含义 |
|------|------|------|
| `raw_pos_mse` | `((logits - smoothed_GT)²)[smoothed_GT > 0.1].mean()` | 行人区域回归精度，floor = 0 |
| `raw_neg_mse` | `((logits - smoothed_GT)²)[smoothed_GT < 0.01].mean()` | 背景抑制效果 |
| `snr` | `logits[pos].mean() - logits[neg].mean()` | 信噪比，越高检测越容易 |

### 2.1 `src/trainer.py` — train_epoch 指标替换

在 `with torch.no_grad()` 监控块中：

1. 用 `map_kernel`（已有参数）对 `pooled_gt` 做 Gaussian 卷积，得到 `smoothed_gt`
2. 计算 `raw_pos_mse`、`raw_neg_mse`、`snr` 替代 `pos_mse`
3. 同步更新 step log 格式和 epoch 返回字典
4. `aux_pos_mse` 同理改为 `aux_raw_pos_mse`

关键代码（替换现有 pos_mse 计算块）：

```python
# 构建 Gaussian-smoothed GT（与 loss 同目标）
B, C, H, W = map_logits.shape
_tgt = pooled_gt.reshape(B * C, 1, H, W)
_k = map_kernel.to(dtype=_tgt.dtype, device=_tgt.device)
_pad = (_k.shape[-1] - 1) // 2
smoothed_gt = F.conv2d(_tgt, _k, padding=_pad).reshape(B, C, H, W)

pos_mask = smoothed_gt > 0.1
neg_mask = smoothed_gt < 0.01
diff2 = (map_logits - smoothed_gt) ** 2

raw_pos_mse = diff2[pos_mask].mean().item() if pos_mask.any() else float("nan")
raw_neg_mse = diff2[neg_mask].mean().item() if neg_mask.any() else float("nan")
snr = (map_logits[pos_mask].mean() - map_logits[neg_mask].mean()).item() if pos_mask.any() and neg_mask.any() else float("nan")
```

### 2.2 `src/trainer.py` — validate() 扩展

validate() 当前只返回 loss/bev_loss/img_loss。添加同样的三个新指标。

### 2.3 `src/train_main.py` — epoch 日志更新

epoch 结束日志中打印新指标（raw_pos_mse / snr）替代原有格式。

---

## 文件变更清单

| 文件 | 变更 | 规模 |
|------|------|------|
| `src/config.py` | +2 行常量 | 极小 |
| `src/train_main.py` | 改默认值引用 + 更新 epoch log | ~6 行 |
| `src/trainer.py` | 替换 pos_mse → raw_pos_mse/snr，扩展 validate() | ~30 行改/增 |
| `scripts/colab_train.py` | 添加 --bev_pos_weight 参数 + 传递 | ~8 行 |
| `.github/workflows/colab-train.yml` | 添加 bev_pos_weight input | ~5 行 |

## 验证方式

1. 本地 `pytest tests/` 通过（确保无语法/导入错误）
2. 推送后触发 Actions 训练（待 Google Drive 配额恢复）
3. 观察新指标：
   - `raw_pos_mse` 应持续下降（无 floor 限制）
   - `snr` 应持续上升（行人信号与背景差距增大）
   - `raw_neg_mse` 应保持低位（背景抑制正常）
4. 训练后自动评估 MODA/MODP（已集成在工作流中）
