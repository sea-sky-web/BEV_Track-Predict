# 评估管线差异分析：MVDet 原始实现 vs 本项目

> **日期**：2026-06-24
> **状态**：已确认，待修复
> **影响**：所有历史 run 的检测指标（F1 0.09–0.12）不反映模型真实能力

## 背景

所有训练 run 的 F1 均在 0.09–0.12，precision ~10%，recall ~14%，接近随机猜测。
但训练指标（loss 持续下降、SNR=0.55）表明模型在学习有效特征。

通过逐行对比 MVDet 原始实现，发现问题不在模型架构或损失函数，而在**评估管线**
的两处关键偏差。

## 参考代码

- **MVDet 论文**：Hou, Zheng, Gould. *Multiview Detection with Feature Perspective
  Transformation*. ECCV 2020.
  [https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520001.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520001.pdf)
- **MVDet 仓库**：[https://github.com/hou-yz/MVDet](https://github.com/hou-yz/MVDet)
- **关键文件对照**：
  - Loss: [`multiview_detector/loss/gaussian_mse.py`](https://github.com/hou-yz/MVDet/blob/master/multiview_detector/loss/gaussian_mse.py)
  - Trainer: [`multiview_detector/trainer.py`](https://github.com/hou-yz/MVDet/blob/master/multiview_detector/trainer.py)
  - NMS: [`multiview_detector/utils/nms.py`](https://github.com/hou-yz/MVDet/blob/master/multiview_detector/utils/nms.py)
  - Model: [`multiview_detector/models/persp_trans_detector.py`](https://github.com/hou-yz/MVDet/blob/master/multiview_detector/models/persp_trans_detector.py)
  - Dataset: [`multiview_detector/datasets/frameDataset.py`](https://github.com/hou-yz/MVDet/blob/master/multiview_detector/datasets/frameDataset.py)
  - Wildtrack: [`multiview_detector/datasets/Wildtrack.py`](https://github.com/hou-yz/MVDet/blob/master/multiview_detector/datasets/Wildtrack.py)

## 已验证一致的部分

以下部分与 MVDet 原始实现一致，**不需要修改**：

| 对比项 | MVDet 原始 | 本项目 | 参考位置 |
|--------|-----------|--------|---------|
| Gaussian 核归一化 | `kernel / kernel.max()` | `g / g.max()` | `frameDataset.py:44` |
| BEV sigma | `20 / grid_reduce = 5.0` | `5.0` | `frameDataset.py:17` |
| 核尺寸 | `2*20+1 = 41` | `41` | `frameDataset.py:40-41` |
| Loss 输入空间 | raw model output | raw logits | `trainer.py:46` |
| GT 构造 | `adaptive_max_pool2d` + `conv2d` | 相同 | `gaussian_mse.py:17-19` |

## 差异 1（致命）：检测阈值的输出空间

### MVDet 做法

模型输出 raw 值（无激活函数），直接用阈值进行检测：

```python
# MVDet trainer.py:55
pred = (map_res > self.cls_thres).int()  # cls_thres=0.4, raw output
```

模型 `forward()` 不含 sigmoid（`persp_trans_detector.py:87`）。训练和推理都在
同一个 raw output 空间操作。

### 本项目做法（有问题）

```python
# evaluate_main.py:249
map_res = torch.sigmoid(map_logits)  # ← 训练时不存在的变换
# 然后 threshold sweep 0.05~0.50 在 sigmoid 输出上
```

### 为什么这是致命的

训练目标是 `MSE(raw_output, gaussian_smoothed_GT)`，GT 峰值 = 1.0，背景 = 0.0。
模型学到的典型 raw output 分布：

- 行人位置：≈ 0.7–0.9
- 背景区域：≈ 0.0–0.1

| 空间 | 行人 | 背景 | 信噪间距 |
|------|------|------|---------|
| Raw logit | 0.80 | 0.05 | **0.75** |
| sigmoid(·) | 0.69 | 0.51 | **0.18** |

sigmoid 将信噪间距从 0.75 压缩到 0.18（压缩 76%）。当 threshold=0.1 时，
背景 sigmoid 值 0.51 远超阈值 → 全图检测点 → 海量 FP。

### 修复

```python
# evaluate_main.py — 去掉 sigmoid，直接用 raw logits
# map_res = torch.sigmoid(map_logits)  # 删除这行
map_res = map_logits                    # 直接使用 raw output
```

阈值改为 0.4（MVDet 默认），或在 0.2–0.6 范围 sweep。

## 差异 2（严重）：NMS 方法和参数

### MVDet 做法

贪心距离抑制，按 score 降序，抑制距离内的较低分检测点：

```python
# MVDet nms.py
def nms(points, scores, dist_thres=50/2.5, top_k=50):
    # dist_thres = 20 grid cells（原始尺度）
    # 在 reduced scale (÷4): 5 cells = 0.5m 物理距离
    # top_k = 50 per frame
```

### 本项目做法（有问题）

```python
# evaluate_main.py:314
pooled = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
keep = (hm >= threshold) & (hm >= pooled - 1e-12)
# max_preds = 200 per frame
```

### 对比

| 参数 | MVDet | 本项目 | 差距 |
|------|-------|--------|------|
| NMS 方法 | 距离贪心抑制 | max_pool2d 局部极值 | 根本不同 |
| 抑制半径 | 5 cells = 0.5m | 1 cell = 0.1m | **5x** |
| 每帧上限 | 50 | 200 | **4x** |

max_pool2d NMS 只保留 3×3 局部极大值，不做更大范围的抑制。Gaussian smoothed GT
的响应区域直径约 20 cells（2σ=10，再加裙摆），3×3 NMS 会在一个行人的响应区域内
保留多个"峰"，产生重复检测。

### 修复

实现 MVDet 风格的距离贪心 NMS，参数：`dist_thres=5`（reduced scale cells），
`top_k=50`。

## 影响评估

这两个差异的组合效应：
1. sigmoid 让**所有像素**都通过低阈值（FP 爆炸）
2. 弱 NMS 无法抑制重复检测（FP 进一步放大）
3. 结果：每帧 ~200 个检测点中绝大多数是 FP → precision ~10%

**模型训练本身没有问题**。修复评估管线后，同一个 checkpoint 应该能产生显著不同
的检测指标。Run #30 的 checkpoint 可以直接用修复后的 eval 重新评估。
